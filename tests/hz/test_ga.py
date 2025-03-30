import numpy as np
import tvm
from tvm import relax

def load_ga_model(lib_path, dev):
    model = tvm.runtime.load_module(lib_path)
    vm = relax.VirtualMachine(model, dev)
    return vm
def test_forward():
    lib_path = "/home/hz/qzq_work/tvm/outputs/ga.so"
    dev = tvm.cuda(0)

    vm = load_ga_model(lib_path, dev)
    forward_func = vm["forward"]
    pop = np.random.randint(0, 2, (1000, 96)).astype(np.int32)
    pop = tvm.nd.array(pop, dev)
    res = forward_func(pop)
    print(res[-1])

def test_ga():
    lib_path = "/home/hz/qzq_work/tvm/outputs/ga.so"
    dev = tvm.cuda(0)
    vm = load_ga_model(lib_path, dev)
    init_func = vm["random_init"]
    forward_func = vm["forward"]
    pop = init_func()
    # pop = tvm.nd.array(pop, dev)
    print("init pop", pop)
    for i in range(1000):
        res = forward_func(pop)
        pop = res[0]
        # print(pop)
        print(f"[{i}]best value{res[-1]}, vars is {res[-2]}")

def test_cross1():
    lib_path = "/home/hz/qzq_work/tvm/outputs/ga.so"
    dev = tvm.cuda(0)
    vm = load_ga_model(lib_path, dev)
    cross_func = vm["debug_cross"]
    init_func = vm["random_init"]
    pop = init_func()
    fitness = np.random.random([10]).astype(np.float32)
    fitness_tvm = tvm.nd.array(fitness, dev)
    out = cross_func(pop, fitness_tvm)
    out_np = fitness /  np.sum(fitness)
    out_np = np.cumsum(out_np)
    breakpoint()
    pass
if __name__ == "__main__":
    test_ga()