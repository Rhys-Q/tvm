import argparse
from tvm import relax
import tvm
from hz.genetic_algorithm import GeneticAlgorithmWarp, GeneticAlgorithmConfig
from tvm.relax.frontend.nn.ga.dispatch_ga import DispatchGACreation


def compile(output_path = "genetic_algorithm.so"):
    config = GeneticAlgorithmConfig()
    model = GeneticAlgorithmWarp(config)

    mod, named_params, ext_mods = model.export_tvm(
        spec=model.get_default_spec(),  # type: ignore
        allow_extern=True,
    )
    # mod = DispatchGACreation()(mod)
    print(mod, flush=True)
    target = tvm.target.Target("llvm")
    exec = relax.build(mod, target=target, params=named_params)

    exec.export_library(output_path)

def run(lib_path):
    model = tvm.runtime.load_module(lib_path)
    vm = relax.VirtualMachine(model, tvm.cpu(0))
    ga = vm["create_genetic_algorithm"]()
    vm.set_input("run", ga)
    vm.invoke_stateful("run")
    out = vm.get_outputs("run")
    print(out)


def test_random(lib_path):
    model = tvm.runtime.load_module(lib_path)
    dev = tvm.cpu(0)
    vm = relax.VirtualMachine(model, dev)
    init_data = vm["random_init"](None)
    print(init_data)
def parse_args():
    parser = argparse.ArgumentParser(description="Compile genetic algorithm model")
    parser.add_argument("--output", type=str, default="genetic_algorithm.so", help="Output file")
    parser.add_argument("--lib_path", type=str, default="genetic_algorithm.so", help="lib path")
    parser.add_argument("--mode", type=str, default="compile", help="mode, compile or run")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "compile":
        compile(args.output)
    elif args.mode == "run":
        run(args.lib_path)
    elif args.mode == "test_random":
        test_random(args.lib_path)
    else:
        raise ValueError("unknown mode: {}".format(args.mode))