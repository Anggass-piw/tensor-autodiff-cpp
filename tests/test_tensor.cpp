#include "tensor.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
int failures = 0;
static_assert(std::is_same_v<decltype(std::declval<tinyad::Tensor&>().values()),
                             const std::vector<double>&>,
              "Tensor::values() must not expose resizable mutable storage");

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

void near(double actual, double expected, const std::string& message) {
    check(std::abs(actual - expected) < 1e-10,
          message + " (expected " + std::to_string(expected) + ", got " + std::to_string(actual) + ")");
}

template <typename Exception, typename Function>
void throws(Function&& function, const std::string& message) {
    try {
        function();
        check(false, message + " (no exception)");
    } catch (const Exception&) {
    } catch (...) {
        check(false, message + " (wrong exception type)");
    }
}

void test_validation() {
    throws<std::invalid_argument>([] { tinyad::Tensor({2, 2}, {1.0, 2.0}); }, "reject data/shape mismatch");
    throws<std::invalid_argument>([] { tinyad::Tensor({2, 0}, 0.0); }, "reject zero-sized dimensions");
    tinyad::Tensor scalar = tinyad::Tensor::scalar(2.5);
    near(scalar.item(), 2.5, "scalar item");
    throws<std::logic_error>([] { tinyad::Tensor({2}, {1.0, 2.0}).item(); }, "item requires one element");

    tinyad::Tensor indexed({2}, {1.0, 2.0});
    indexed.flat_at(1) = 4.0;
    near(indexed.values()[1], 4.0, "flat_at mutates an element without changing storage size");
    throws<std::out_of_range>([&] { (void)indexed.flat_at(indexed.size()); },
                              "reject flat index past tensor storage");
}

void test_leaf_element_updates() {
    auto leaf = tinyad::Value::variable(tinyad::Tensor({2}, {1.0, 2.0}));
    leaf.mutable_data(1) = 3.0;
    near(leaf.data().values()[1], 3.0, "update one leaf element for an optimizer step");
    throws<std::out_of_range>([&] { (void)leaf.mutable_data(leaf.data().size()); },
                              "reject leaf update past tensor storage");

    auto result = leaf.sum();
    throws<std::logic_error>([&] { (void)result.mutable_data(0); },
                             "reject in-place element updates on non-leaf values");
}

void test_elementwise_and_shared_graph() {
    auto x = tinyad::Value::variable(tinyad::Tensor({2}, {1.0, 2.0}));
    auto y = tinyad::Value::variable(tinyad::Tensor({2}, {3.0, 4.0}));
    auto loss = (x * y + x).sum();
    near(loss.data().item(), 14.0, "elementwise expression forward");
    loss.backward();
    near(x.grad().values()[0], 4.0, "elementwise gradient x[0]");
    near(x.grad().values()[1], 5.0, "elementwise gradient x[1]");
    near(y.grad().values()[0], 1.0, "elementwise gradient y[0]");
    near(y.grad().values()[1], 2.0, "elementwise gradient y[1]");

    auto z = tinyad::Value::variable(tinyad::Tensor::scalar(3.0));
    auto repeated = (z * z + z).sum();
    repeated.backward();
    near(z.grad().item(), 7.0, "gradient accumulation across a shared node");
    repeated.backward();
    near(z.grad().item(), 14.0, "repeated backward accumulates leaf gradients once per call");
}

void test_scalar_ops_relu_and_shapes() {
    auto x = tinyad::Value::variable(tinyad::Tensor({3}, {-2.0, 0.0, 3.0}));
    auto loss = ((x * 2.0).relu()).sum();
    loss.backward();
    near(x.grad().values()[0], 0.0, "ReLU gradient below zero");
    near(x.grad().values()[1], 0.0, "ReLU gradient at zero");
    near(x.grad().values()[2], 2.0, "scalar multiplication and ReLU gradient");
    throws<std::invalid_argument>([] {
        auto a = tinyad::Value::constant(tinyad::Tensor({2}, 1.0));
        auto b = tinyad::Value::constant(tinyad::Tensor({1, 2}, 1.0));
        (void)(a + b);
    }, "reject implicit broadcasting");
    throws<std::logic_error>([] {
        auto vector = tinyad::Value::variable(tinyad::Tensor({2}, 1.0));
        vector.backward();
    }, "require scalar output for backward");
}

void test_matmul() {
    auto a = tinyad::Value::variable(tinyad::Tensor({2, 2}, {1.0, 2.0, 3.0, 4.0}));
    auto b = tinyad::Value::variable(tinyad::Tensor({2, 2}, {5.0, 6.0, 7.0, 8.0}));
    auto product_sum = a.matmul(b).sum();
    near(product_sum.data().item(), 134.0, "matmul forward and reduction");
    product_sum.backward();
    const double expected_a[] = {11.0, 15.0, 11.0, 15.0};
    const double expected_b[] = {4.0, 4.0, 6.0, 6.0};
    for (std::size_t i = 0; i < 4; ++i) {
        near(a.grad().values()[i], expected_a[i], "matmul left gradient " + std::to_string(i));
        near(b.grad().values()[i], expected_b[i], "matmul right gradient " + std::to_string(i));
    }
    throws<std::invalid_argument>([] {
        auto lhs = tinyad::Value::constant(tinyad::Tensor({2, 3}, 1.0));
        auto rhs = tinyad::Value::constant(tinyad::Tensor({2, 2}, 1.0));
        (void)lhs.matmul(rhs);
    }, "reject incompatible matmul inner dimensions");
}
} // namespace

int main() {
    test_validation();
    test_leaf_element_updates();
    test_elementwise_and_shared_graph();
    test_scalar_ops_relu_and_shapes();
    test_matmul();
    if (failures != 0) {
        std::cerr << failures << " test(s) failed\n";
        return 1;
    }
    std::cout << "All tensor/autodiff tests passed.\n";
    return 0;
}
