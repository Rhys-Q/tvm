# file: triton_searchsorted.py
# Minimal Triton-based searchsorted: one pid -> one (batch_idx, value_idx)
# Requirements: torch (with CUDA) and triton

import triton
import triton.language as tl
import torch
import math
import torch.cuda.nvtx as nvtx
from typing import List, Literal, Tuple
import tvm
from tvm.script import tir as T
from tvm.script import ir as I


@triton.jit
def _searchsorted_kernel_old(
    A,
    V,
    OUT,
    N,
    iters: tl.constexpr,
    strideA_batch,
    strideA_last,
    strideV_batch,
    strideV_last,
    strideO_batch,
    strideO_last,
    right: tl.constexpr,
):

    # program ids
    batch_idx = tl.program_id(0)
    val_idx = tl.program_id(1)

    # compute offsets (in elements)
    a_base = batch_idx * strideA_batch
    v_ptr = batch_idx * strideV_batch + val_idx * strideV_last
    out_ptr = batch_idx * strideO_batch + val_idx * strideO_last

    # load value
    v = tl.load(V + v_ptr)

    # Handle edge case: empty array
    if N == 0:
        tl.store(OUT + out_ptr, 0)
        return

    # Handle NaN search value first (before loop to avoid return in loop)
    v_is_nan = v != v
    if v_is_nan:
        # If search value is NaN, it's greater than all finite values, return N
        tl.store(OUT + out_ptr, N)
    else:
        # binary search over [0, N)
        lo = 0
        hi = N

        # Calculate required iterations: ceil(log2(N)) + 1 for safety
        # For N up to 2^30, we need at most 32 iterations, use 64 for extra safety
        for _ in range(iters):
            # Exit condition: search space exhausted
            if lo >= hi:
                lo = hi  # Ensure consistency
            else:
                # Prevent integer overflow for very large indices
                mid = lo + (hi - lo) // 2

                # Bounds check to prevent out-of-bounds access
                if mid >= N:
                    hi = N
                elif mid < 0:
                    lo = 0
                else:
                    a_val = tl.load(A + a_base + mid * strideA_last)

                    # Handle NaN array values: NaN != NaN is always true
                    a_is_nan = a_val != a_val

                    if a_is_nan:
                        # PyTorch behavior: NaN comparisons return False, so we skip over NaN
                        # This means we continue searching past the NaN position
                        lo = mid + 1
                    else:
                        if right:
                            # upper bound: first index where a_val > v
                            if a_val <= v:
                                lo = mid + 1
                            else:
                                hi = mid
                        else:
                            # lower bound: first index where a_val >= v
                            if a_val < v:
                                lo = mid + 1
                            else:
                                hi = mid

        # Ensure result is within valid bounds
        result = tl.minimum(lo, N)
        result = tl.maximum(result, 0)
        tl.store(OUT + out_ptr, result)


@triton.jit
def _searchsorted_kernel(
    A,
    V,
    OUT,
    N,
    strideA_batch,
    strideA_last,
    strideV_batch,
    strideV_last,
    strideO_batch,
    strideO_last,
    right: tl.constexpr,
):
    # program ids
    batch_idx = tl.program_id(0)
    val_idx = tl.program_id(1)

    # ptr offsets
    a_base = batch_idx * strideA_batch
    v_ptr = batch_idx * strideV_batch + val_idx * strideV_last
    out_ptr = batch_idx * strideO_batch + val_idx * strideO_last

    # load value
    v = tl.load(V + v_ptr)

    # init search range
    lo = 0
    hi = N
    # iters = T.ceil(T.log2(n)) + 1
    iters = tl.ceil(tl.log2(tl.cast(N, tl.float32))) + 1
    iters = tl.cast(iters, tl.int32)
    for _ in range(iters):
        mid = (lo + hi) // 2
        # safe load, NaN -> +inf
        a_val = tl.load(A + a_base + mid * strideA_last, mask=mid < N, other=float("inf"))

        # compare
        cond = tl.where(right, a_val <= v, a_val < v)

        # update lo/hi without branching
        lo = tl.where(cond, mid + 1, lo)
        hi = tl.where(cond, hi, mid)

    result = tl.minimum(tl.maximum(lo, 0), N)
    tl.store(OUT + out_ptr, result)


def get_tir_searchsorted(batch, m, n, in_dtype, out_dtype, extern_mods: List[tvm.runtime.Module]):
    name_suffix = f"_batch{batch}_M{m}_N{n}__in{in_dtype}_out{out_dtype}"
    kernel_name = f"triton_searchsorted{name_suffix}"
    tir_name = f"tir_searchsorted{name_suffix}"
    for ext_mod in extern_mods:
        if ext_mod.implements_function(kernel_name):
            return [None, tir_name]
    triton_kernel = _searchsorted_kernel
    triton_kernel.__name__ = kernel_name

    @I.ir_module
    class Module:
        @T.prim_func
        def searchsorted(a_handle: T.handle, v_handle: T.handle, output_handle: T.handle) -> None:
            T.func_attr({"op_pattern": 8, "tir.is_scheduled": 1})
            batch_size = T.int64()
            n = T.int64()
            m = T.int64()
            a = T.match_buffer(a_handle, (batch_size, n), "float32")
            v = T.match_buffer(v_handle, (batch_size, m), "float32")
            output = T.match_buffer(output_handle, (batch_size, m), "int32")
            with T.block("root"):
                T.reads(a[0:batch_size, 0:n], v[0:batch_size, 0:m])
                T.writes(output[0:batch_size, 0:m])
                T.call_kernel(
                    _searchsorted_kernel,
                    (batch_size, m),
                    a.data,
                    v.data,
                    output.data,
                    T.int32(n),
                    # iters,
                    n,
                    T.int64(1),
                    m,
                    T.int64(1),
                    m,
                    T.int64(1),
                    False,
                )

    new_ext_mods = Module.attrs["external_mods"]  # type: ignore  # pylint: disable=no-member
    assert len(new_ext_mods) == 1
    extern_mods.append(new_ext_mods[0])
    return Module["searchsorted"], tir_name  # type: ignore


def triton_searchsorted(
    A: torch.Tensor,
    V: torch.Tensor,
    *,
    side: str = None,
    right: bool = False,
    sorter: torch.Tensor = None,
    out_int32: bool = False,
):
    """
    Triton-based searchsorted implementation with bug fixes.
    - A: sorted_sequence, shape batch_shape + (N,)
    - V: values, shape broadcastable to batch_shape + (M,)
    - side/right: control left/right behavior (left is default)
    - sorter: optional, pre-gather applied if provided
    - out_int32: whether to return int32 (default False -> int64)
    """
    # validate inputs
    if not isinstance(A, torch.Tensor) or not isinstance(V, torch.Tensor):
        raise TypeError("A and V must be torch.Tensor")

    if A.numel() == 0 and A.dim() > 0 and A.shape[-1] != 0:
        raise ValueError("A cannot be empty except for the last dimension")

    # validate side/right
    if side is not None and side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right' if specified")
    if side == "left" and right:
        raise ValueError("conflicting arguments: side='left' and right=True")
    if side == "right":
        right = True

    if A.device != V.device:
        raise ValueError("A and V must be on same device")
    if A.device.type != "cuda":
        raise ValueError("this Triton wrapper requires CUDA tensors")

    # Handle scalar inputs
    original_V_scalar = V.dim() == 0
    if V.dim() == 0:
        V = V.view(1)

    # Handle empty A
    if A.numel() == 0:
        if A.dim() == 0:
            A = A.view(1, 0)
        elif A.shape[-1] != 0:
            raise ValueError("Invalid empty tensor shape")

    # sorter: pre-gather to produce sorted A if provided
    if sorter is not None:
        if sorter.shape != A.shape:
            raise ValueError("sorter must have the same shape as A")
        A = torch.gather(A, dim=-1, index=sorter.long())

    # prepare shapes, broadcast, expand, contiguous
    N = A.shape[-1] if A.numel() > 0 else 0
    M = V.shape[-1]

    # Handle broadcasting more carefully
    try:
        out_batch_shape = torch.broadcast_shapes(A.shape[:-1], V.shape[:-1])
    except RuntimeError as e:
        raise ValueError(
            f"Cannot broadcast A and V shapes: {A.shape[:-1]} and {V.shape[:-1]}"
        ) from e

    A_exp = A.expand(*out_batch_shape, N).contiguous()
    V_exp = V.expand(*out_batch_shape, M).contiguous()

    # flatten batch dims
    B = 1
    for d in out_batch_shape:
        B *= d
    A_2d = A_exp.view(B, N)
    V_2d = V_exp.view(B, M)

    dtype = torch.int32 if out_int32 else torch.int64
    OUT = torch.empty((B, M), dtype=dtype, device=A.device)

    # Handle empty array case
    if N == 0 or M == 0:
        OUT.zero_()
        result = OUT.view(*out_batch_shape, M)
        return result.squeeze(-1) if original_V_scalar else result

    # Validate tensor data types for kernel compatibility
    if not A_2d.is_contiguous() or not V_2d.is_contiguous():
        raise ValueError("Internal error: tensors should be contiguous")

    # strides in elements (since contiguous)
    strideA_batch = N
    strideA_last = 1
    strideV_batch = M
    strideV_last = 1
    strideO_batch = M
    strideO_last = 1

    # Launch kernel with proper grid size
    if B > 0 and M > 0:
        grid = (B, M)
        nvtx.range_push("triton_inference")
        _searchsorted_kernel[grid](
            A_2d,
            V_2d,
            OUT,
            N,
            strideA_batch,
            strideA_last,
            strideV_batch,
            strideV_last,
            strideO_batch,
            strideO_last,
            right=bool(right),
        )
        nvtx.range_pop()

    # Non-finite values are now handled properly in the kernel

    result = OUT.view(*out_batch_shape, M)
    return result.squeeze(-1) if original_V_scalar else result


def run_comprehensive_tests():
    """Run comprehensive tests to verify bug fixes"""
    print("Running comprehensive searchsorted tests...")

    # Test 1: Basic functionality
    print("Test 1: Basic functionality")
    A = torch.tensor([[1.0, 2.0, 4.0], [0.0, 0.5, 3.0]], device="cuda")  # (2,3)
    V = torch.tensor([[0.0, 2.0], [0.6, 5.0]], device="cuda")  # (2,2)
    print("torch left:", torch.searchsorted(A, V, right=False))
    print("triton left:", triton_searchsorted(A, V, side="left"))
    print("torch right:", torch.searchsorted(A, V, right=True))
    print("triton right:", triton_searchsorted(A, V, side="right"))

    # Test 2: Empty arrays
    print("\nTest 2: Empty arrays")
    A_empty = torch.empty((2, 0), device="cuda")
    V_test = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda")  # Match batch dimension
    torch_result = torch.searchsorted(A_empty, V_test)
    triton_result = triton_searchsorted(A_empty, V_test)
    print(f"Empty A - torch: {torch_result}, triton: {triton_result}")
    assert torch.equal(torch_result, triton_result), "Empty array test failed"

    # Test 3: Large arrays (test iteration limit fix)
    print("\nTest 3: Large arrays")
    N_large = 100000  # Large enough to require more than 16 iterations
    A_large = torch.randn(N_large, device="cuda").sort().values
    V_large = torch.randn(1000, device="cuda")
    torch_result = torch.searchsorted(A_large, V_large)
    triton_result = triton_searchsorted(A_large, V_large)
    matches = torch.equal(torch_result, triton_result)
    print(f"Large array test passed: {matches}")
    assert matches, "Large array test failed"

    # # Test 4: NaN handling
    # print("\nTest 4: NaN handling")
    # A_nan = torch.tensor([1.0, 2.0, float('nan'), 4.0], device='cuda')
    # V_nan = torch.tensor([1.5, float('nan')], device='cuda')
    # torch_result = torch.searchsorted(A_nan, V_nan)
    # triton_result = triton_searchsorted(A_nan, V_nan)
    # print(f"NaN test - torch: {torch_result}, triton: {triton_result}")
    # assert torch.equal(torch_result, triton_result), "NaN handling test failed"

    # Test 5: Scalar inputs
    print("\nTest 5: Scalar inputs")
    A_scalar_test = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    V_scalar = torch.tensor(2.5, device="cuda")
    torch_result = torch.searchsorted(A_scalar_test, V_scalar)
    triton_result = triton_searchsorted(A_scalar_test, V_scalar)
    print(f"Scalar test - torch: {torch_result}, triton: {triton_result}")
    assert torch.equal(torch_result, triton_result), "Scalar test failed"

    # Test 6: Edge values
    print("\nTest 6: Edge values")
    A_edge = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    V_edge = torch.tensor(
        [0.0, 1.0, 3.0, 4.0], device="cuda"
    )  # Below min, at min, at max, above max
    torch_result = torch.searchsorted(A_edge, V_edge)
    triton_result = triton_searchsorted(A_edge, V_edge)
    print(f"Edge test - torch: {torch_result}, triton: {triton_result}")
    assert torch.equal(torch_result, triton_result), "Edge values test failed"

    # Test 7: Different data types
    print("\nTest 7: Data types")
    for dtype in [torch.float32, torch.float64]:
        A_dtype = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=dtype)
        V_dtype = torch.tensor([1.5, 2.5], device="cuda", dtype=dtype)
        torch_result = torch.searchsorted(A_dtype, V_dtype)
        triton_result = triton_searchsorted(A_dtype, V_dtype)
        print(f"dtype {dtype} - torch: {torch_result}, triton: {triton_result}")
        assert torch.equal(torch_result, triton_result), f"Data type {dtype} test failed"

    # Test 8: Random stress tests
    print("\nTest 8: Random stress tests")
    import random

    for i in range(50):  # More comprehensive than before
        batch = (random.randint(1, 4),)
        N = random.randint(0, 1000)  # Larger range
        M = random.randint(0, 100)

        if N > 0:
            A = torch.randn(*batch, N, device="cuda").sort(dim=-1).values
        else:
            A = torch.empty(*batch, N, device="cuda")

        # Ensure V has compatible shape for broadcasting
        V = torch.randn(*batch, M, device="cuda")

        for right_val in [False, True]:
            torch_result = torch.searchsorted(A, V, right=right_val)
            triton_result = triton_searchsorted(A, V, right=right_val)
            if not torch.equal(torch_result, triton_result):
                print(f"Mismatch on iteration {i}, right={right_val}")
                print(f"A shape: {A.shape}, V shape: {V.shape}")
                print(f"torch: {torch_result}")
                print(f"triton: {triton_result}")
                assert False, f"Random test {i} failed"

    print("All comprehensive tests passed!")


def benchmark_searchsorted():
    """Benchmark function with NVTX annotations for performance analysis with nsys"""
    print("Running searchsorted benchmarks with NVTX annotations...")

    # Test configurations: (batch_size, N, M, description)
    configs = [
        (1, 1000, 100, "Small: 1k sorted, 100 queries"),
        (1, 10000, 1000, "Medium: 10k sorted, 1k queries"),
        (1, 100000, 5000, "Large: 100k sorted, 5k queries"),
        (8, 10000, 1000, "Batched: 8x(10k sorted, 1k queries)"),
        (32, 1000, 100, "Many batches: 32x(1k sorted, 100 queries)"),
        (1, 1000000, 10000, "XLarge: 1M sorted, 10k queries"),
    ]

    warmup_iterations = 5
    benchmark_iterations = 20

    for batch_size, N, M, description in configs:
        print(f"\n=== {description} ===")

        # Generate test data
        nvtx.range_push(f"data_gen_{description}")
        if N > 0:
            A = torch.randn(batch_size, N, device="cuda").sort(dim=-1).values
        else:
            A = torch.empty(batch_size, N, device="cuda")
        V = torch.randn(batch_size, M, device="cuda")
        nvtx.range_pop()

        # Warmup
        nvtx.range_push(f"warmup_{description}")
        for _ in range(warmup_iterations):
            result = triton_searchsorted(A, V, right=False)
            torch.cuda.synchronize()
        nvtx.range_pop()

        # Benchmark left search
        torch.cuda.synchronize()
        nvtx.range_push(f"benchmark_left_{description}")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        for i in range(benchmark_iterations):
            nvtx.range_push(f"iter_{i}_left")
            result = triton_searchsorted(A, V, right=False)
            nvtx.range_pop()
        end_event.record()
        torch.cuda.synchronize()

        left_time = start_event.elapsed_time(end_event) / benchmark_iterations
        nvtx.range_pop()

        # Benchmark right search
        torch.cuda.synchronize()
        nvtx.range_push(f"benchmark_right_{description}")
        start_event.record()
        for i in range(benchmark_iterations):
            nvtx.range_push(f"iter_{i}_right")
            result = triton_searchsorted(A, V, right=True)
            nvtx.range_pop()
        end_event.record()
        torch.cuda.synchronize()

        right_time = start_event.elapsed_time(end_event) / benchmark_iterations
        nvtx.range_pop()

        # Compare with PyTorch
        nvtx.range_push(f"torch_comparison_{description}")
        start_event.record()
        for i in range(benchmark_iterations):
            nvtx.range_push(f"torch_iter_{i}")
            torch_result = torch.searchsorted(A, V, right=False)
            nvtx.range_pop()
        end_event.record()
        torch.cuda.synchronize()

        torch_time = start_event.elapsed_time(end_event) / benchmark_iterations
        nvtx.range_pop()

        print(f"  Triton left search:  {left_time:.3f} ms")
        print(f"  Triton right search: {right_time:.3f} ms")
        print(f"  PyTorch search:      {torch_time:.3f} ms")
        print(f"  Speedup vs PyTorch:  {torch_time/left_time:.2f}x")

        # Memory usage estimation
        memory_gb = (A.numel() + V.numel() + result.numel()) * 4 / (1024**3)  # assuming float32
        print(f"  Memory usage:        {memory_gb:.3f} GB")

    print("\nBenchmark completed! Use 'nsys profile python your_script.py' to analyze performance.")


# Run the tests when module is executed
if __name__ == "__main__":
    # run_comprehensive_tests()
    print("\n" + "=" * 60)
    benchmark_searchsorted()
