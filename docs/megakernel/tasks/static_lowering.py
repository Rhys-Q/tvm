# constants: M, N, K (matrix dimensions)
# BLK_M, BLK_N (block size), WORLD_SIZE(number of devices)
M = 1024
N = 1024
K = 256
WORLD_SIZE=1
LOCAL_M = M // WORLD_SIZE
BLK_M = 16
BLK_N = 16
######################################################################
# before lowering
######################################################################
@device_func
def matmul(i: int, j: int, A: Tensor, B: Tensor, C: Tensor):
    with cta():
        matmul_cta(C[i*BLK_M:(i+1)*BLK_M, j*BLK_N: (j+1)*BLK_N],A[i*BLK_M:(i+1)*BLK_M,:], B[:, j*BLK_N:(j+1)*BLK_N])

@device_func
def reduce_scatter(i: int, j: int, C: Tensor, D: Tensor):
    with cta():
        offset = get_rank() * LOCAL_M
        
        multimem_ld_reduce_cta(D[i*BLK_M:(i+1)*BLK_M, j*BLK_N: (j+1)*BLK_N], C[offset+i*BLK_M: offset+(i+1)*BLK_M, j*BLK_N: (j+1)*BLK_N])

@graph_func
def main_graph(A: Tensor((M, K)), B: Tensor((K,N))) -> Tensor((LOCAL_M, N)):
    E = ETensor((M // BLK_M, N // BLK_N), wait_count = WORLD_SIZE, shard="S[0]")
    C: Tensor((M, N)) = call_device(matmul, tile_num = (M // BLK_M, N // BLK_N), args = [A,B], in_edges={}, out_edges={E: "ij->ij"})
    
    q = E.local_view()
    D: Tensor((LOCAL_M, N)) = call_device(reduce_scatter, tile_num=(LOCAL_M // BLK_M, N // BLK_N), args=[C], in_edges={E_local: "ij->ij"}, out_edges={})
    
    return D

######################################################################
# after lowering
######################################################################
# fuse with explicit event notify and wait
@device_func
def fused_matmul_rs(sm_id: int, A: Tensor, B: Tensor, C: Tensor, E: ETensor, D: Tensor):
    # sm_id is the task coordinate
    with cta():
        tile_scheduler = init_tile_scheduler(sm_id)
        while tile_scheduler.vaild():
            task_idx, task_type = tile_scheduler.get_task()
            if task_type == 0:
                i, j = task_idx
                matmul_cta(C[...], A[...], B[...])
                E[i, j].notify()
            else:
                i, j = task_idx
                offset = get_rank() * LOCAL_M
                E[i+offset // BLK_M, j].wait()
                
                multimem_ld_reduce_cta(D[...], C[...])
            tile_scheduler.next_tile()

# fuse to a persistent device function call with SM_COUNT tiles
@device_func
def main_graph(
    A: Tensor((M, K)), B: Tensor((K, N))
) -> Tensor((LOCAL_M, N)):
    E = ETensor((M // BLK_M, N // BLK_N), wait_count = WORLD_SIZE, shard="S[0]")
    C: Tensor((M,N)) = empty((M,N))
    
    E_local = E.local_view()
    D: Tensor((LOCAL_M, N)) = call_device(fused_matmul_rs, tile_num=(SM_COUNT), args=[A, B, C, E], in_edges={}, out_edges={})
    
    return D