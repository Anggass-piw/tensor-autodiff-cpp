#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace tinyad {

// An owning, row-major, contiguous CPU tensor of doubles. A shape of {} is a scalar.
class Tensor {
public:
    Tensor(std::vector<std::size_t> shape, std::vector<double> values);
    explicit Tensor(std::vector<std::size_t> shape, double fill_value = 0.0);

    static Tensor scalar(double value);

    const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    const std::vector<double>& values() const noexcept { return values_; }
    std::size_t size() const noexcept { return values_.size(); }
    double item() const;
    double& flat_at(std::size_t index);
    const double& flat_at(std::size_t index) const;
    double at2d(std::size_t row, std::size_t col) const;
    double& at2d(std::size_t row, std::size_t col);
    Tensor zeros_like() const;

private:
    std::vector<std::size_t> shape_;
    std::vector<double> values_;
};

struct Node {
    Tensor data;
    Tensor grad;
    bool requires_grad;
    bool is_leaf;
    std::vector<std::shared_ptr<Node>> parents;
    std::function<void()> backward_fn;

    Node(Tensor value, bool tracks_grad, bool leaf);
};

// A small reverse-mode differentiable value. Gradients are accumulated by backward().
class Value {
public:
    static Value variable(Tensor data);
    static Value constant(Tensor data);

    const Tensor& data() const noexcept;
    // Mutable element access is intended for optimizer updates on leaf values only.
    double& mutable_data(std::size_t flat_index);
    const Tensor& grad() const noexcept;
    bool requires_grad() const noexcept;
    void zero_grad();
    void backward(); // The output must contain exactly one element.

    Value sum() const;
    Value relu() const;
    Value matmul(const Value& rhs) const; // 2-D matrix multiplication.

    friend Value operator+(const Value& lhs, const Value& rhs);
    friend Value operator*(const Value& lhs, const Value& rhs); // Elementwise multiplication.
    friend Value operator+(const Value& lhs, double rhs);
    friend Value operator+(double lhs, const Value& rhs);
    friend Value operator*(const Value& lhs, double rhs);
    friend Value operator*(double lhs, const Value& rhs);

private:
    explicit Value(std::shared_ptr<Node> node) : node_(std::move(node)) {}
    std::shared_ptr<Node> node_;
};

} // namespace tinyad
