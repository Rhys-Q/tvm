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
    best_value = 100000000
    best_vars = None
    best_value_unchange_cnt = 0
    for i in range(100):
        res = forward_func(pop)
        pop = res[0]
        local_best_value = res[-1].numpy().tolist()[0][0]
        if local_best_value < best_value:
            best_value = local_best_value
            best_vars = res[-2].numpy().tolist()
            best_value_unchange_cnt = 0
        else:
            best_value_unchange_cnt += 1
        print(f"[{i}]best value {best_value}, vars is {best_vars}")
        if best_value_unchange_cnt > 100:
            pop = init_func()
            best_value_unchange_cnt = 0
            # print("reset pop", flush=True)


def test_cross():
    lib_path = "/home/hz/qzq_work/tvm/outputs/ga.so"
    dev = tvm.cuda(0)
    vm = load_ga_model(lib_path, dev)
    cross_func = vm["debug_cross"]
    num_cross = 400
    pop = np.random.randint(0, 2, (1000, 96)).astype(np.int32)
    fitness = np.random.random([1000]).astype(np.float32)
    cross_val = np.random.random([num_cross,1 ]).astype(np.float32)
    cross_index = np.random.randint(0, 24, (num_cross//2, 4, 24)).astype(np.int32)
    tvm_inputs = [pop, fitness,cross_val,cross_index]
    tvm_inputs = [tvm.nd.array(i, dev) for i in tvm_inputs]
    out_tvm = cross_func(*tvm_inputs)
    out_tvm = out_tvm.numpy()
    print(out_tvm)


    def cross_numpy(pop, fitness, cross_val, cross_index):
        # pop [num_pop, 96]
        # fitness [num_pop]
        # cross_val [num_cross, 1]
        # cross_index [num_cross//2, 4, 24]
        num_pop = pop.shape[0]
        num_var = 4
        pop = np.reshape(pop, [ num_pop, num_var, -1]) # [1000, 4, 24]
        sum_fitness = np.sum(fitness) # [1000]
        fitness = fitness / sum_fitness
        fitness = np.cumsum(fitness)
        flag1 = cross_val <= fitness # [800, 1000]
        flag1 = np.astype(flag1, "uint8")
        selected_index_1 = np.argmax(flag1, axis=1) #[800]
        cross = np.take(pop, selected_index_1, axis=0)  #[800, 4, 24]
        cross_1, cross_2 = np.split(cross, 2, axis=0)

        cross_pop1 = np.where(cross_index > 0, cross_1, cross_2)
        cross_pop2 = np.where(cross_index <= 0, cross_1, cross_2)
        cross_pop = np.concat([cross_pop1, cross_pop2], axis=0) # [800, 4, 24]
        cross_pop = np.reshape(cross_pop, [cross_pop.shape[0], -1]) # [800, 96]
        return cross_pop
    np_out = cross_numpy(pop, fitness, cross_val, cross_index)
    np.testing.assert_allclose(out_tvm, np_out, rtol=1e-05, atol=1e-05)


if __name__ == "__main__":
    test_cross()