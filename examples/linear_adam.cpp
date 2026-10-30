#include "tensor.hpp"

#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // Fit y = 2x + 1 with scalar parameters, using four deterministic examples.
    const std::vector<double> inputs{-1.0, 0.0, 1.0, 2.0};
    const std::vector<double> targets{-1.0, 1.0, 3.0, 5.0};
    auto weight = tinyad::Value::variable(tinyad::Tensor::scalar(0.0));
    auto bias = tinyad::Value::variable(tinyad::Tensor::scalar(0.0));

    double first_moment_w = 0.0, second_moment_w = 0.0;
    double first_moment_b = 0.0, second_moment_b = 0.0;
    constexpr double learning_rate = 0.05;
    constexpr double beta1 = 0.9;
    constexpr double beta2 = 0.999;
    constexpr double epsilon = 1e-8;
    constexpr int steps = 1200;

    for (int step = 1; step <= steps; ++step) {
        weight.zero_grad();
        bias.zero_grad();
        auto total_loss = tinyad::Value::constant(tinyad::Tensor::scalar(0.0));
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const auto x = tinyad::Value::constant(tinyad::Tensor::scalar(inputs[i]));
            const auto y = tinyad::Value::constant(tinyad::Tensor::scalar(targets[i]));
            const auto error = weight * x + bias + y * -1.0;
            total_loss = total_loss + error * error;
        }
        auto mean_squared_error = total_loss * (1.0 / static_cast<double>(inputs.size()));
        mean_squared_error.backward();

        const double gradient_w = weight.grad().item();
        const double gradient_b = bias.grad().item();
        first_moment_w = beta1 * first_moment_w + (1.0 - beta1) * gradient_w;
        second_moment_w = beta2 * second_moment_w + (1.0 - beta2) * gradient_w * gradient_w;
        first_moment_b = beta1 * first_moment_b + (1.0 - beta1) * gradient_b;
        second_moment_b = beta2 * second_moment_b + (1.0 - beta2) * gradient_b * gradient_b;

        const double correction1 = 1.0 - std::pow(beta1, step);
        const double correction2 = 1.0 - std::pow(beta2, step);
        weight.mutable_data(0) -= learning_rate * (first_moment_w / correction1) /
            (std::sqrt(second_moment_w / correction2) + epsilon);
        bias.mutable_data(0) -= learning_rate * (first_moment_b / correction1) /
            (std::sqrt(second_moment_b / correction2) + epsilon);
    }

    std::cout << "Fit: weight=" << weight.data().item() << ", bias=" << bias.data().item() << '\n';
    return (std::abs(weight.data().item() - 2.0) < 1e-3 &&
            std::abs(bias.data().item() - 1.0) < 1e-3) ? 0 : 1;
}
