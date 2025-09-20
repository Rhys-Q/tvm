# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Apache TVM is a compiler stack for deep learning systems designed to bridge productivity-focused deep learning frameworks and performance-focused hardware backends. The current architecture features:

- **TensorIR (TIR)**: Tensor-level intermediate representation for low-level optimizations
- **Relax**: Graph-level representation for high-level program structure and transformations
- **Python-first design**: Most transformations are customizable in Python
- **Cross-level optimization**: Joint optimization of computational graphs, tensor programs, and libraries

## Build System

### Primary Build Commands
```bash
# Configure and build (creates build/ directory)
make

# Build specific targets
make runtime     # Build runtime only
make cpptest     # Build C++ tests
make crttest     # Build CRT tests

# Alternative CMake approach
mkdir build && cd build
cmake ..
make
```

### Configuration
- Main config: `cmake/config.cmake` (copy to root or build/ directory to customize)
- Build directory: `build/` (configurable via `TVM_BUILD_PATH`)
- Key CMake options include hardware backends (CUDA, OpenCL, Vulkan, ROCM, Metal)

## Development Commands

### Linting and Formatting
```bash
# Comprehensive linting
make lint            # Runs cpplint, pylint, jnilint
make cpplint         # C++ linting only
make pylint          # Python linting only

# Code formatting
make format          # Auto-format with clang-format, black, cargo fmt

# Type checking
make mypy            # Run MyPy type checking
tests/scripts/task_mypy.sh
```

### Testing
```bash
# Python tests (use pytest with custom wrapper)
cd tests/python && python -m pytest [path/to/test]

# C++ tests (after building with cpptest target)
./build/cpptest

# Specific test suites via scripts
tests/scripts/task_python_unittest.sh
tests/scripts/task_cpp_unittest.sh
tests/scripts/task_python_integration.sh
```

### Python Development
```bash
# Set up environment
export TVM_PATH=$(pwd)
export PYTHONPATH="${TVM_PATH}/python"

# Install Python dependencies
pip install -e .                    # Basic installation
pip install -e ".[dev]"            # With development tools
pip install -e ".[all]"            # With all optional dependencies
```

## Code Architecture

### Core Directories

**C++ Source (`src/`)**:
- `runtime/`: Core runtime system and device management
- `target/`: Hardware target definitions and code generation
- `tir/`: TensorIR implementation and transformations
- `relax/`: Relax IR implementation and graph-level optimizations
- `arith/`: Arithmetic analysis and simplification
- `node/`: AST node system and object model
- `meta_schedule/`: Auto-scheduling and tuning framework

**Python Interface (`python/tvm/`)**:
- `tir/`: TensorIR Python bindings and utilities
- `relax/`: Relax Python interface and transformations
- `meta_schedule/`: Auto-scheduling Python interface
- `runtime/`: Runtime Python bindings
- `target/`: Target specification utilities
- `script/`: TVMScript for writing TIR/Relax programs
- `contrib/`: Integration with external frameworks

**Other Key Directories**:
- `include/`: C++ headers and public APIs
- `apps/`: Example applications and benchmarks
- `tests/`: Comprehensive test suite (Python and C++)
- `docs/`: Documentation source
- `3rdparty/`: External dependencies
- `ffi/`: Foreign Function Interface implementations

### Key Architectural Patterns

1. **Hybrid C++/Python Architecture**: Core optimizations in C++ with Python bindings for usability
2. **Node System**: Immutable AST nodes with copy-on-write semantics
3. **Pass Framework**: Composable transformation passes for IR optimization
4. **Target System**: Hardware-specific code generation and optimization
5. **PackedFunc**: Unified function calling convention across languages

### Language Bindings
- **Python**: Primary interface (`python/tvm/`)
- **Java**: JVM bindings (`jvm/`)
- **JavaScript**: WebAssembly support (`web/`)
- **Rust**: Rust bindings (`rust/`)

## Testing Strategy

### Test Organization
- `tests/python/`: Python unit and integration tests
- `tests/cpp/`: C++ unit tests
- `tests/scripts/`: CI/CD automation scripts
- Test execution uses custom pytest wrapper with sharding support

### Running Tests
Use the `run_pytest` function from `tests/scripts/setup-pytest-env.sh` for consistent test execution with proper environment setup and result reporting.

## Development Tools Configuration

### Python Tools
- **Black**: Line length 100, excludes build artifacts and 3rdparty
- **MyPy**: Type checking enabled for core packages (tir/schedule, meta_schedule, etc.)
- **pytest**: Configured with reruns, parallel execution, and JUnit XML output
- **isort**: Black-compatible import sorting

### Build Requirements
- CMake 3.18+
- Python 3.9+
- Optional: CUDA, OpenCL, Vulkan, ROCM for hardware-specific features

# import tvm的方法
import tvm前需要设置PYTHONPATH，如下所示
export PYTHONPATH=/root/brother/tvm/python 