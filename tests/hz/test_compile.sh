#!/bin/bash

export PYTHONPATH=/home/hz/qzq_work/tvm/python


output_path=/home/hz/qzq_work/tvm/outputs/ga.so
# python /home/hz/qzq_work/tvm/python/hz/compile.py --mode compile --output $output_path

# python /home/hz/qzq_work/tvm/python/hz/compile.py --mode run --lib_path $output_path

# python /home/hz/qzq_work/tvm/python/hz/compile.py --mode test_random --lib_path $output_path

python /home/hz/qzq_work/tvm/python/hz/compile.py --mode test_gpu_compile --output $output_path

# python /home/hz/qzq_work/tvm/python/hz/compile.py --mode test_random_cuda --lib_path $output_path