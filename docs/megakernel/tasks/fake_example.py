class IRModule:
    
    @device_func
    def partial_sum(i: int, j: int, A: Tensor, B: Tensor):
        B[i*32: i*32 + 32, j] = sum(A[i*32: i*32 + 32, j*32:j*32+32])
    
    @device_func
    def final_sum(i: int, B: Tensor, C: Tensor):
        C[i*32: i*32+32] = sum(B[i*32: i*32+32, :])
    
    @graph_func
    def main_graph(A: Tensor(("n*32", 128))) -> Tensor(("n*32",)):
        n = sym_var()
        E = ETensor((n,), wait_count = 4)
        B :Tensor((n*32, 4)) = call_device(IRModule.partial_sum, tile_num=(n,4), args=[A], in_edge={}, out_edges={E: "ij->i"},)
        
        C : Tensor((n*32,)) = call_device(IRModule.final_sum, tile_num=(n,), args=[B], in_edges={E:"i->i"}, out_edges = {},)
        return C