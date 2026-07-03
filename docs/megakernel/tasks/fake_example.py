from tvm.script import tirx as T
from tvm.tirx.lang import (
    ETensor,
    Tensor,
    call_device,
    device_func,
    graph_func,
    lower_event_tensor_graph,
)

ROW_TILE = 32
N_COLS = 128
K_PARTS = 4


class IRModule:
    @device_func
    def partial_sum(i, j, A, B, tx):
        if tx < ROW_TILE:
            row = i * ROW_TILE + tx
            acc = T.float32(0)
            for c in T.serial(N_COLS // K_PARTS):
                acc = acc + A[row, j * (N_COLS // K_PARTS) + c]
            B[row, j] = acc

    @device_func
    def final_sum(i, B, C, tx):
        if tx < ROW_TILE:
            row = i * ROW_TILE + tx
            acc = T.float32(0)
            for j in T.serial(K_PARTS):
                acc = acc + B[row, j]
            C[row] = acc

    @graph_func
    def main_graph(A: Tensor, n_tiles: int) -> Tensor:
        E = ETensor((n_tiles,), wait_count=K_PARTS)
        B: Tensor = call_device(
            IRModule.partial_sum,
            tile_num=(n_tiles, K_PARTS),
            args=[A],
            outputs=Tensor((n_tiles * ROW_TILE, K_PARTS)),
            in_edges={},
            out_edges={E: "ij->i"},
            threads=ROW_TILE,
        )
        C: Tensor = call_device(
            IRModule.final_sum,
            tile_num=(n_tiles,),
            args=[B],
            outputs=Tensor((n_tiles * ROW_TILE,)),
            in_edges={E: "i->i"},
            out_edges={},
            threads=ROW_TILE,
        )
        return C


def trace_graph(n_tiles: int = 1):
    return IRModule.main_graph(Tensor((n_tiles * ROW_TILE, 128)), n_tiles)


def build_static(n_tiles: int = 1):
    return lower_event_tensor_graph(trace_graph(n_tiles), schedule="static")


def build_dynamic(n_tiles: int = 1):
    return lower_event_tensor_graph(trace_graph(n_tiles), schedule="dynamic")
