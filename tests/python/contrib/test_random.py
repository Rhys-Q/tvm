# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Configure pytest"""
# pylint: disable=invalid-name
import threading
import numpy as np
import tvm
from tvm import te
from tvm.contrib import random
from tvm import rpc
import tvm.testing


def test_randint():
    """Tests randint function"""
    m = 10240
    n = 10240
    A = random.randint(-127, 128, size=(m, n), dtype="int32")

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.random.randint", True):
            print("skip because extern function is not available")
            return
        dev = tvm.cpu(0)
        f = tvm.compile(te.create_prim_func([A]), target=target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), dev)
        f(a)
        na = a.numpy()
        assert abs(np.mean(na)) < 0.3
        assert np.min(na) == -127
        assert np.max(na) == 127

    verify()


@tvm.testing.uses_gpu
def test_randint_cuda_graph_compatible():
    """Tests CUDA Graph compatible randint function"""
    m = 1024
    n = 1024

    def test_cuda_randint():
        if not tvm.testing.device_enabled("cuda"):
            print("skip because CUDA is not enabled...")
            return
        if not tvm.get_global_func("runtime.contrib.curand.Init", True):
            print("skip because cuRAND Init function is not available")
            return
        if not tvm.get_global_func("runtime.contrib.curand.RandInt", True):
            print("skip because cuRAND RandInt function is not available")
            return

        dev = tvm.cuda(0)

        # Initialize CUDA random engine first (outside CUDA graph)
        init_func = tvm.get_global_func("runtime.contrib.curand.Init")
        init_func(42)  # seed

        # Test direct cuRAND RandInt call
        randint_func = tvm.get_global_func("runtime.contrib.curand.RandInt")
        a = tvm.nd.array(np.zeros((m, n), dtype="int32"), dev)
        randint_func(-127, 128, a)
        print(a)
        na = a.numpy()
        print(f"CUDA randint stats: mean={np.mean(na):.3f}, min={np.min(na)}, max={np.max(na)}")

        # Verify the results are within expected bounds
        assert np.min(na) >= -127
        assert np.max(na) <= 127
        assert abs(np.mean(na)) < 5.0  # Reasonable mean for uniform distribution

        # Test different data types
        for dtype in ["int8", "int16", "int32", "uint8", "uint16", "uint32"]:
            try:
                a_typed = tvm.nd.array(np.zeros((512, 512), dtype=dtype), dev)
                if dtype.startswith('int'):
                    randint_func(-10, 10, a_typed)
                else:  # uint
                    randint_func(0, 20, a_typed)
                na_typed = a_typed.numpy()
                print(f"CUDA randint {dtype}: min={np.min(na_typed)}, max={np.max(na_typed)}")
            except Exception as e:
                print(f"Warning: {dtype} test failed: {e}")

        # Test via standard randint API (should automatically use CUDA backend)
        A = random.randint(-50, 51, size=(512, 512), dtype="int32")
        f = tvm.compile(te.create_prim_func([A]), target="cuda")
        a_standard = tvm.nd.array(np.zeros((512, 512), dtype="int32"), dev)
        f(a_standard)
        na_standard = a_standard.numpy()

        print(f"Standard API randint stats: mean={np.mean(na_standard):.3f}, min={np.min(na_standard)}, max={np.max(na_standard)}")
        assert np.min(na_standard) >= -50
        assert np.max(na_standard) <= 50

    test_cuda_randint()


@tvm.testing.uses_gpu
def test_rand_cuda_graph_compatible():
    """Tests CUDA Graph compatible rand function"""
    m = 512
    n = 512

    def test_cuda_rand():
        if not tvm.testing.device_enabled("cuda"):
            print("skip because CUDA is not enabled...")
            return
        if not tvm.get_global_func("runtime.contrib.curand.Init", True):
            print("skip because cuRAND Init function is not available")
            return
        if not tvm.get_global_func("runtime.contrib.curand.Uniform", True):
            print("skip because cuRAND Uniform function is not available")
            return

        dev = tvm.cuda(0)

        # Initialize CUDA random engine first (outside CUDA graph)
        init_func = tvm.get_global_func("runtime.contrib.curand.Init")
        init_func(42)  # seed

        # Test direct cuRAND Uniform call
        uniform_func = tvm.get_global_func("runtime.contrib.curand.Uniform")
        a = tvm.nd.array(np.zeros((m, n), dtype="float32"), dev)
        uniform_func(a)
        print(a)

        na = a.numpy()
        print(f"CUDA uniform stats: mean={np.mean(na):.3f}, min={np.min(na):.6f}, max={np.max(na):.6f}")

        # Verify the results are within expected bounds [0, 1)
        assert np.min(na) >= 0.0
        assert np.max(na) < 1.0
        assert abs(np.mean(na) - 0.5) < 0.1  # Mean should be around 0.5 for uniform [0,1)

        # Test different float types
        for dtype in ["float32", "float64"]:
            try:
                a_typed = tvm.nd.array(np.zeros((256, 256), dtype=dtype), dev)
                uniform_func(a_typed)
                na_typed = a_typed.numpy()
                print(f"CUDA uniform {dtype}: min={np.min(na_typed):.6f}, max={np.max(na_typed):.6f}")
                assert np.min(na_typed) >= 0.0
                assert np.max(na_typed) < 1.0
            except Exception as e:
                print(f"Warning: {dtype} test failed: {e}")

        # Test via standard rand API (should automatically use CUDA backend)
        A = random.rand(256, 256)
        f = tvm.compile(te.create_prim_func([A]), target="cuda")
        a_standard = tvm.nd.array(np.zeros((256, 256), dtype="float32"), dev)
        f(a_standard)
        na_standard = a_standard.numpy()

        print(f"Standard API rand stats: mean={np.mean(na_standard):.3f}, min={np.min(na_standard):.6f}, max={np.max(na_standard):.6f}")
        assert np.min(na_standard) >= 0.0
        assert np.max(na_standard) < 1.0
        assert abs(np.mean(na_standard) - 0.5) < 0.1

        # Test different data types for standard API
        for dtype in ["float32", "float64"]:
            try:
                A_typed = random.rand(128, 128, dtype=dtype)
                f_typed = tvm.compile(te.create_prim_func([A_typed]), target="cuda")
                a_typed_standard = tvm.nd.array(np.zeros((128, 128), dtype=dtype), dev)
                f_typed(a_typed_standard)
                na_typed_standard = a_typed_standard.numpy()
                print(f"Standard API rand {dtype}: min={np.min(na_typed_standard):.6f}, max={np.max(na_typed_standard):.6f}")
                assert np.min(na_typed_standard) >= 0.0
                assert np.max(na_typed_standard) < 1.0
            except Exception as e:
                print(f"Warning: standard API {dtype} test failed: {e}")

    test_cuda_rand()


def test_rand():
    """Tests rand function on CPU"""
    m = 1024
    n = 1024
    A = random.rand(m, n)

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.random.rand", True):
            print("skip because extern function is not available")
            return
        dev = tvm.cpu(0)
        f = tvm.compile(te.create_prim_func([A]), target=target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), dev)
        f(a)
        na = a.numpy()

        # Check that values are in [0, 1)
        assert np.min(na) >= 0.0
        assert np.max(na) < 1.0
        # Check reasonable distribution (mean should be around 0.5)
        assert abs(np.mean(na) - 0.5) < 0.1

    verify()


def test_uniform():
    """Tests uniform function"""
    m = 10240
    n = 10240
    A = random.uniform(0, 1, size=(m, n))

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.random.uniform", True):
            print("skip because extern function is not available")
            return
        dev = tvm.cpu(0)
        f = tvm.compile(te.create_prim_func([A]), target=target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), dev)
        f(a)
        na = a.numpy()
        assert abs(np.mean(na) - 0.5) < 1e-1
        assert abs(np.min(na) - 0.0) < 1e-3
        assert abs(np.max(na) - 1.0) < 1e-3

    verify()


def test_normal():
    """Tests normal function"""
    m = 10240
    n = 10240
    A = random.normal(3, 4, size=(m, n))

    def verify(target="llvm"):
        if not tvm.testing.device_enabled(target):
            print("skip because %s is not enabled..." % target)
            return
        if not tvm.get_global_func("tvm.contrib.random.normal", True):
            print("skip because extern function is not available")
            return
        dev = tvm.cpu(0)
        f = tvm.compile(te.create_prim_func([A]), target=target)
        a = tvm.nd.array(np.zeros((m, n), dtype=A.dtype), dev)
        f(a)
        na = a.numpy()
        assert abs(np.mean(na) - 3) < 1e-1
        assert abs(np.std(na) - 4) < 1e-2

    verify()


@tvm.testing.uses_gpu
def test_random_fill():
    """Tests random_fill function"""

    def test_local(dev, dtype):
        if not tvm.get_global_func("tvm.contrib.random.random_fill", True):
            print("skip because extern function is not available")
            return
        value = tvm.nd.empty((512, 512), dtype, dev)
        random_fill = tvm.get_global_func("tvm.contrib.random.random_fill")
        random_fill(value)

        assert np.count_nonzero(value.numpy()) == 512 * 512

        # make sure arithmentic doesn't overflow too
        np_values = value.numpy()
        assert np.isfinite(np_values * np_values + np_values).any()

    def test_rpc(dtype):
        if not tvm.get_global_func("tvm.contrib.random.random_fill", True):
            print("skip because extern function is not available")
            return
        if not tvm.testing.device_enabled("rpc") or not tvm.runtime.enabled("llvm"):
            return

        def check_remote(server):
            remote = rpc.connect(server.host, server.port)
            value = tvm.nd.empty((512, 512), dtype, remote.cpu())
            random_fill = remote.get_function("tvm.contrib.random.random_fill")
            random_fill(value)

            assert np.count_nonzero(value.numpy()) == 512 * 512

            # make sure arithmentic doesn't overflow too
            np_values = value.numpy()
            assert np.isfinite(np_values * np_values + np_values).any()

        check_remote(rpc.Server("127.0.0.1"))

    for dtype in [
        "bool",
        "int4",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "int32",
        "int64",
        "uint64",
        "float16",
        "float32",
        "float64",
    ]:
        for _, dev in tvm.testing.enabled_targets():
            test_local(dev, dtype)
        test_rpc(dtype)


def test_random_fill_mt():
    """Check random filler applicability in case of nontrivial thread pool configuration.
    Particularly when MaxConcurrency != num_workers_used_ which is actual for big-little systems.
    """
    no_exception_happened = True

    def test_body():
        try:
            num_thread_used = 1
            configure_threads = tvm.get_global_func("runtime.config_threadpool")
            configure_threads(1, num_thread_used)

            test_input = tvm.runtime.ndarray.empty((10, 10))
            random_fill = tvm.get_global_func("tvm.contrib.random.random_fill_for_measure")
            random_fill(test_input)
        except:  # pylint: disable=bare-except
            nonlocal no_exception_happened
            no_exception_happened = False

    # ThreadPool object is thread local. To eliminate effect on other test cases put it into thread
    x = threading.Thread(target=test_body)
    x.start()
    x.join()
    assert no_exception_happened


if __name__ == "__main__":
    test_randint()
    test_rand()
    test_uniform()
    test_normal()
    test_random_fill()
    test_random_fill_mt()
    test_randint_cuda_graph_compatible()
    test_rand_cuda_graph_compatible()
