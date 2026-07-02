# constants: M, N, K (matrix dimensions)
# BLK_M, BLK_N (block size)
M = 1024
N = 1024
K = 256
BLK_M = 16
BLK_N = 16

######################################################################
# before lowering
######################################################################
@device_func
def matmul(i: int, j: int, A: Tensor, B: Tensor, C: Tensor):
    with cta():
        matmul_cta(
            C[i * BLK_M : (i + 1) * BLK_M, j * BLK_N : (j + 1) * BLK_N],
            A[i * BLK_M : (i + 1) * BLK_M, :],
            B[:, j * BLK_N : (j + 1) * BLK_N],
        )


@device_func
def epilogue(i: int, j: int, C: Tensor, D: Tensor):
    with cta():
        store_cta(
            D[i * BLK_M : (i + 1) * BLK_M, j * BLK_N : (j + 1) * BLK_N],
            C[i * BLK_M : (i + 1) * BLK_M, j * BLK_N : (j + 1) * BLK_N],
        )


@graph_func
def main_graph(A: Tensor((M, K)), B: Tensor((K, N))) -> Tensor((M, N)):
    E = ETensor((M // BLK_M, N // BLK_N), wait_count=1)
    C: Tensor((M, N)) = call_device(
        matmul,
        tile_num=(M // BLK_M, N // BLK_N),
        args=[A, B],
        in_edges={},
        out_edges={E: "ij->ij"},
    )
    D: Tensor((M, N)) = call_device(
        epilogue,
        tile_num=(M // BLK_M, N // BLK_N),
        args=[C],
        in_edges={E: "ij->ij"},
        out_edges={},
    )

    return D


######################################################################
# after lowering
######################################################################
# fuse with explicit event notify and wait
@device_func
def fused_matmul_epilogue(
    sm_id: int, A: Tensor, B: Tensor, C: Tensor, E: ETensor, D: Tensor
):
    # sm_id is the task coordinate
    with cta():
        tile_scheduler = init_tile_scheduler(sm_id)
        while tile_scheduler.valid():
            task_idx, task_type = tile_scheduler.get_task()
            if task_type == 0:
                i, j = task_idx
                matmul_cta(C[...], A[...], B[...])
                E[i, j].notify()
            else:
                i, j = task_idx
                E[i, j].wait()
                store_cta(D[...], C[...])
            tile_scheduler.next_tile()


# fuse to a persistent device function call with SM_COUNT tiles
@device_func
def main_graph(A: Tensor((M, K)), B: Tensor((K, N))) -> Tensor((M, N)):
    E = ETensor((M // BLK_M, N // BLK_N), wait_count=1)
    C: Tensor((M, N)) = empty((M, N))
    D: Tensor((M, N)) = call_device(
        fused_matmul_epilogue,
        tile_num=(SM_COUNT),
        args=[A, B, C, E],
        in_edges={},
        out_edges={},
    )

    return D
