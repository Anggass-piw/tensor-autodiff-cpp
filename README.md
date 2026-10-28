# Tiny Tensor Autodiff (C++17)

A small, dependency-free CPU tensor and reverse-mode automatic differentiation MVP. It is intended as an understandable starting point for experimentation, not as a production-grade framework or a replacement for PyTorch.

## Build and test

Requirements: a C++17 compiler and CMake 3.16 or newer. No third-party libraries are required.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

The build creates `linear_adam`, a runnable training demonstration. Run it directly with:

```sh
./build/linear_adam
```

It fits the scalar linear model `y = weight * x + bias` with a small Adam implementation and exits unsuccessfully if the learned parameters are not close to 2 and 1.

## API at a glance

Include `tensor.hpp` and link the `tinyad` CMake target. `Tensor` owns contiguous row-major `double` data. Construct it with a shape and matching values, or use a fill value; `{}` denotes a scalar.

```cpp
#include "tensor.hpp"

using tinyad::Tensor;
using tinyad::Value;

auto x = Value::variable(Tensor({2}, {1.0, 2.0}));
auto y = Value::constant(Tensor({2}, {3.0, 4.0}));
auto loss = (x * y + x).sum(); // elementwise multiply, exact-shape add, reduce
loss.backward();
// x.grad() is [4, 5]
```

The available differentiable operations are exact-shape addition, exact-shape elementwise multiplication, scalar addition and multiplication, full sum reduction, ReLU, and 2-D matrix multiplication. `backward()` seeds a one-element output with gradient 1; reduce a multi-element output with `sum()` first. Gradients accumulate, so call `zero_grad()` on leaf variables before a new optimization step. Optimizers can update leaf data through `mutable_data()`.

## Current MVP boundaries

- CPU only; values are `double` and stored contiguously in row-major order.
- Shapes must match the supplied element count. Dimensions must be positive; the empty shape represents one scalar.
- Elementwise operations require identical shapes. Broadcasting, slicing, views, and general indexing are not implemented.
- Matrix multiplication accepts only rank-2 inputs.
- The graph is built from the listed operations and supports first-order reverse-mode gradients. There is no higher-order differentiation, graph release policy, checkpointing, or in-place graph mutation support.
- `backward()` supports scalar outputs only; use `sum()` to reduce first.
- No serialization, device abstraction, SIMD/kernel tuning, threading, sparse tensors, or neural-network module library is included. The example's Adam implementation is demonstration code, not part of the library API.

## Roadmap

1. Add broadcasting with well-defined shape inference and gradient reduction tests.
2. Add general reductions, transpose/reshape views, and broader indexing semantics.
3. Improve graph lifetime and gradient-reset ergonomics; add finite-difference gradient checks.
4. Add batched matmul and benchmark-driven CPU kernel improvements.
5. Build higher-level layers and optimizers only after the tensor and gradient contracts are stable.

Contributions should preserve the small dependency-free build and add tests for forward values, invalid shapes, and gradients for each new operation.
