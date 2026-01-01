import argparse
import random
import time
import os
import torch
import torch.distributed as dist
import numpy as np
from functools import partial
from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from flashinfer.cute_dsl.utils import (
    get_cutlass_dtype,
    get_num_sm,
    is_cute_dsl_available,
)
from flashinfer.cute_dsl.blockscaled_gemm import (
    Sm100BlockScaledPersistentDenseGemmKernel,  # not used in python interface
    grouped_gemm_nt_masked,  # deepgemm-like python interface for DLFW integration
    create_scale_factor_tensor,
)

import deep_ep
from utils import init_dist, bench, bench_kineto, calc_diff, hash_tensor, per_token_cast_back

from vllm.model_executor.layers.fused_moe.flashinfer_cutedsl_moe import flashinfer_cutedsl_moe_masked
from vllm.model_executor.layers.fused_moe.deepep_ll_prepare_finalize import (
    CombineOverlapArgs,
    W2GemmOverlapArgs,
)


FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

def produce_nvfp4_weights_and_scales(n, k, l=32):    
    ab_dtype = "float4_e2m1fn"
    b_major = "k"
    device='cuda'
    sf_vec_size = 16
    sf_dtype = "float8_e4m3fn"
    b_ref = cutlass_torch.matrix(
        l, n, k, b_major == "n", cutlass.Float32, device=device
    )
    b_tensor, b_torch = cutlass_torch.cute_tensor_like(
        b_ref,
        get_cutlass_dtype(ab_dtype),
        is_dynamic_layout=True,
        assumed_align=16,
    )
    n, k, l = b_torch.shape
    # slice into half after flatten        
    half_len_b = b_torch.numel() // 2
    b_torch = (
        b_torch.permute(2, 0, 1)
        .flatten()[:half_len_b]
        .reshape(l, n, k // 2)
        .permute(1, 2, 0)
    )
    print("before create scale factor")
    sfb_ref, sfb_tensor, sfb_torch = create_scale_factor_tensor(
        l, n, k, sf_vec_size, get_cutlass_dtype(sf_dtype), device
    )
    print("after create scale factor")
    return b_torch, sfb_torch

def test_main(num_tokens: int, hidden: int, num_experts: int, num_topk: int,
              rank: int, num_ranks: int, group: dist.ProcessGroup, buffer: deep_ep.Buffer,
              use_logfmt: bool = False, seed: int = 0, args: argparse.Namespace = None):

    total_num_sms = torch.cuda.get_device_properties(
        device="cuda"
    ).multi_processor_count
    communicate_num_sms = 32
    compute_num_sms = total_num_sms - communicate_num_sms

    torch.manual_seed(seed + rank)
    random.seed(seed + rank)

    assert num_experts % num_ranks == 0
    num_local_experts = num_experts // num_ranks

    w13 = torch.load("/home/mxz/mylogs/nvfp4-tensors/w1.pt", map_location="cuda")
    w13_sf = torch.load("/home/mxz/mylogs/nvfp4-tensors/w1_scale.pt", map_location="cuda")
    w2 = torch.load("/home/mxz/mylogs/nvfp4-tensors/w2.pt", map_location="cuda")
    w2_sf = torch.load("/home/mxz/mylogs/nvfp4-tensors/w2_scale.pt", map_location="cuda")
    w1_alpha = torch.load("/home/mxz/mylogs/nvfp4-tensors/g1_alphas.pt", map_location="cuda")[:num_local_experts]
    w2_alpha = torch.load("/home/mxz/mylogs/nvfp4-tensors/g2_alphas.pt", map_location="cuda")[:num_local_experts]
    a1_global_scale = torch.load("/home/mxz/mylogs/nvfp4-tensors/a1_gscale.pt", map_location="cuda")[:num_local_experts]
    a2_global_scale = torch.load("/home/mxz/mylogs/nvfp4-tensors/a2_gscale.pt", map_location="cuda")[:num_local_experts]
    workspace = torch.empty((num_local_experts, num_ranks * num_tokens, 4096), dtype=torch.bfloat16, device='cuda')
    output = torch.empty((num_local_experts, num_ranks * num_tokens, 7168), dtype=torch.bfloat16, device='cuda')
    fi_prof_buf1 = torch.zeros(num_local_experts, dtype=torch.int64, device='cuda')
    fi_prof_buf2 = torch.empty((num_local_experts + 1, compute_num_sms, 2), dtype=torch.int64, device='cuda')

    # NOTES: the integers greater than 256 exceed the BF16 precision limit
    rank_offset = 128
    assert num_ranks - rank_offset < 257, 'Too many ranks (exceeding test precision limit)'

    x = torch.ones((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * (rank - rank_offset)
    x[:, -128:] = torch.arange(num_tokens, device='cuda').to(torch.bfloat16).view(-1, 1)
    x_list = [x]
    for i in range(4 if use_logfmt else 0):
        # NOTES: make more LogFMT casts and also with some BF16
        x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.5 * random.random())
    # NOTES: the last one is for performance testing
    # Most of the values in the perf case is lower than the threshold, casting most channels
    x_list.append(torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device='cuda') * 0.1)

    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device='cuda').abs() + 1
    topk_idx = torch.topk(scores, num_topk, dim=-1, largest=True, sorted=True)[1]
    topk_weights = torch.randn((num_tokens, num_topk), dtype=torch.float32, device='cuda').abs()

    # Randomly mask some positions
    #for i in range(10):
    #    topk_idx[random.randint(0, num_tokens - 1), random.randint(0, num_topk - 1)] = -1

    # Check dispatch correctness    
    hash_value, num_times = 0, 0

    current_x = x_list[-1]    
    cumulative_local_expert_recv_stats = torch.zeros((num_local_experts, ), dtype=torch.int, device='cuda')

    alt_stream = torch.cuda.Stream()

    # noinspection PyShadowingNames
    def test_func_nvfp4_baseline(return_recv_hook: bool):
        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=False, use_nvfp4=True, x_global_scale=a1_global_scale,
                                        async_finish=False, return_recv_hook=return_recv_hook)                                        
        hook() if return_recv_hook else None        
        x = recv_x[0].permute(2, 0, 1)
        x_scale = recv_x[1]
        flashinfer_cutedsl_moe_masked(
            hidden_states=(x, x_scale),
            input_global_scale=None,
            w1=w13,
            w1_blockscale=w13_sf,
            w1_alpha=w1_alpha,
            w2=w2,
            a2_global_scale=a2_global_scale,
            w2_blockscale=w2_sf,
            w2_alpha=w2_alpha,
            masked_m=recv_count,
            workspace=workspace,
            out=output,
            w2_gemm_overlap_args=None,
        )        

        combined_x, event, hook = buffer.low_latency_combine(output, topk_idx, topk_weights, handle,
                                                             use_logfmt=use_logfmt, return_recv_hook=return_recv_hook)
        hook() if return_recv_hook else None

    def test_func_nvfp4_sbo(return_recv_hook: bool):
        w2_gemm_overlap_args = None                

        combine_wait_event = torch.cuda.Event()
        combine_overlap_args = CombineOverlapArgs(
            num_sms=communicate_num_sms,
            stream=alt_stream,
            wait_event=combine_wait_event,
        )

        combine_signal = torch.zeros(
            num_local_experts, dtype=torch.uint32, device="cuda"
        )

        w2_gemm_overlap_args = W2GemmOverlapArgs(
            signal=combine_signal,
            start_event=combine_wait_event,
            num_sms=compute_num_sms,
        )
        combine_overlap_args.signal = combine_signal
        combine_overlap_args.threshold = compute_num_sms        

        recv_x, recv_count, handle, event, hook = \
            buffer.low_latency_dispatch(current_x, topk_idx, num_tokens, num_experts,
                                        cumulative_local_expert_recv_stats=cumulative_local_expert_recv_stats,
                                        use_fp8=False, use_nvfp4=True, x_global_scale=a1_global_scale,
                                        async_finish=False, return_recv_hook=return_recv_hook)                                                                                    
        hook() if return_recv_hook else None                         
        x = recv_x[0].permute(2, 0, 1)
        x_scale = recv_x[1]        
        flashinfer_cutedsl_moe_masked(
            hidden_states=(x, x_scale),
            input_global_scale=None,
            w1=w13,
            w1_blockscale=w13_sf,
            w1_alpha=w1_alpha,
            w2=w2,
            a2_global_scale=a2_global_scale,
            w2_blockscale=w2_sf,
            w2_alpha=w2_alpha,
            masked_m=recv_count,
            workspace=workspace,
            out=output,
            w2_gemm_overlap_args=w2_gemm_overlap_args,
            #fi_prof_buf1=fi_prof_buf1,
            #fi_prof_buf2=fi_prof_buf2,
        )

        combine_overlap_args.stream.wait_event(
            combine_overlap_args.wait_event
        )        
        with torch.cuda.stream(combine_overlap_args.stream):
            combined_x, event, hook = buffer.low_latency_combine(
                output, topk_idx, topk_weights, handle,
                use_logfmt=use_logfmt, return_recv_hook=return_recv_hook,
                overlap=True,
                src_signals=combine_overlap_args.signal,
                src_signal_expect_value=combine_overlap_args.threshold,
            )
        hook() if return_recv_hook else None

        torch.cuda.current_stream().wait_stream(combine_overlap_args.stream)

    ########################################################
    # nvfp4
    ########################################################
    # Calculate bandwidth
    num_nvfp4_bytes, num_bf16_bytes = (hidden / 2 + hidden / 16 + 16), hidden * 2
    num_logfmt10_bytes = hidden * 10 / 8 + hidden / 128 * 4
    num_dispatch_comm_bytes, num_combine_comm_bytes = 0, 0
    for i in range(num_tokens):
        num_selections = (topk_idx[i] != -1).sum().item()
        num_dispatch_comm_bytes += num_nvfp4_bytes * num_selections
        num_combine_comm_bytes += (num_logfmt10_bytes if use_logfmt else num_bf16_bytes) * num_selections

        # Dispatch + combine testing    
    avg_t, min_t, max_t = bench(partial(test_func_nvfp4_baseline, return_recv_hook=True), num_tests=1000)
    print(f'[rank {rank}] nvfp4 Dispatch + bf16 combine bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
          f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)
    
    # SBO Dispatch + combine testing
    num_tests = 1000
    avg_t, min_t, max_t = bench(partial(test_func_nvfp4_sbo, return_recv_hook=True), num_tests=num_tests)
    print(f'[rank {rank}] nvfp4 Dispatch + bf16 combine_v2 bandwidth: {(num_dispatch_comm_bytes + num_combine_comm_bytes) / 1e9 / avg_t:.2f} GB/s, '
          f'avg_t={avg_t * 1e6:.2f} us, min_t={min_t * 1e6:.2f} us, max_t={max_t * 1e6:.2f} us', flush=True)

    """        
    if rank == 0:
        fi_prof_buf1 = fi_prof_buf1.cpu()
        for i in range(num_local_experts):
            print(f"rank={rank}, local_expert_id={i}, execution_time={fi_prof_buf1[i] / 1000.0 / num_tests}us")
        event_list = []
        fi_prof_buf2 = fi_prof_buf2.cpu()
        for eidx in range(num_local_experts + 1):
            for sm_id in range(compute_num_sms):
                event_list.append((fi_prof_buf2[eidx, sm_id, 0], eidx, sm_id, fi_prof_buf2[eidx, sm_id, 1]))
        event_list.sort()
        prev_timestamp = 0
        for timestamp, eidx, sm_count, sm_id in event_list:
            if sm_id > 256:
                sm_id = sm_id - (1 << 16)
                record_at_exit = True
            else:
                record_at_exit = False
            print(f"rank={rank}, local_expert_id={eidx}, sm_arrived_count={sm_count}, sm_id={sm_id}, record_at_exit={record_at_exit}, timestamp={timestamp}, delta={timestamp - prev_timestamp}")
            prev_timestamp = timestamp    
    """
    if args.collect_traces:
        bench_kineto(partial(test_func_nvfp4_baseline, return_recv_hook=True),
            kernel_names=('dispatch', 'combine'), barrier_comm_profiling=False,
            suppress_kineto_output=True, num_kernels_per_period=2, num_tests=100,
            trace_path=f"/home/mxz/mylogs/deepep-sbo-traces/nvfp4_baseline_rank{rank}.json.gz")

        bench_kineto(partial(test_func_nvfp4_sbo, return_recv_hook=True),
            kernel_names=('dispatch', 'combine'), barrier_comm_profiling=False,
            suppress_kineto_output=True, num_kernels_per_period=2, num_tests=100,
            trace_path=f"/home/mxz/mylogs/deepep-sbo-traces/nvfp4_sbo_rank{rank}.json.gz")


    return hash_value


# noinspection PyUnboundLocalVariable,PyShadowingNames
def test_loop(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    num_tokens, hidden = args.num_tokens, args.hidden
    num_topk, num_experts = args.num_topk, args.num_experts

    num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(num_tokens, hidden, num_ranks, num_experts)
    if local_rank == 0:
        print(f'Allocating buffer size: {num_rdma_bytes / 1e6} MB ...', flush=True)
    buffer = deep_ep.Buffer(group, num_rdma_bytes=num_rdma_bytes, low_latency_mode=True,
                            num_qps_per_rank=num_experts // num_ranks,
                            allow_nvlink_for_low_latency_mode=not args.disable_nvlink, explicitly_destroy=True,
                            allow_mnnvl=args.allow_mnnvl)
    test_main(num_tokens, hidden, num_experts, num_topk, rank, num_ranks, group, buffer,
              use_logfmt=args.use_logfmt, seed=1, args=args)

    # Destroy the buffer runtime and communication group
    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # TODO: you may modify NUMA binding for less CPU overhead
    # TODO: buggy with `num_tokens=512`
    parser = argparse.ArgumentParser(description='Test low-latency EP kernels')
    parser.add_argument('--num-processes', type=int, default=8,
                       help='Number of processes to spawn (default: 8)')
    parser.add_argument('--num-tokens', type=int, default=128,
                       help='Number of tokens (default: 128)')
    parser.add_argument('--hidden', type=int, default=7168,
                       help='Hidden dimension size (default: 7168)')
    parser.add_argument('--num-topk', type=int, default=8,
                       help='Number of top-k experts (default: 8)')
    parser.add_argument('--num-experts', type=int, default=288,
                       help='Number of experts (default: 288)')
    parser.add_argument('--allow-mnnvl', action="store_true",
                        help='Allow MNNVL for communication')
    parser.add_argument('--disable-nvlink', action='store_true',
                        help='Whether to disable NVLink for testing')
    parser.add_argument('--use-logfmt', action='store_true',
                        help='Whether to test LogFMT combine')
    parser.add_argument("--pressure-test", action='store_true',
                        help='Whether to do pressure test')
    parser.add_argument("--no-kineto-profile", action='store_true',
                        help='Whether to do torch profile')
    parser.add_argument("--collect-traces", action='store_true',
                        help='Whether to do torch profile to collect trace')

    args = parser.parse_args()

    num_processes = args.num_processes
    torch.multiprocessing.spawn(test_loop, args=(num_processes, args), nprocs=num_processes)
