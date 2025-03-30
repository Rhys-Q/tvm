import numpy as np
import tvm
from tvm import relax

def load_ga_model(lib_path = "/home/hz/qzq_work/tvm/outputs/ga.so"):
    model = tvm.runtime.load_module(lib_path)
    dev = tvm.cuda(0)
    vm = relax.VirtualMachine(model, dev)
    return vm
def test_forward():
    vm = load_ga_model()
    dev = tvm.cuda(0)
    forward_func = vm["forward"]
    pop = np.random.randint(0, 2, (1000, 96)).astype(np.int32)
    pop = tvm.nd.array(pop, dev)
    res = forward_func(pop)
    print(res[-1])

def test_ga():
    vm = load_ga_model()
    dev = tvm.cuda(0)
    forward_func = vm["forward"]
    pop = np.random.randint(0, 2, (1000, 96)).astype(np.int32)
    pop = tvm.nd.array(pop, dev)
    for i in range(10000):
        res = forward_func(pop)
        pop = res[0]
        print(f"[{i}]best value{res[-1]}, vars is {res[-2]}")


if __name__ == "__main__":
    test_ga()