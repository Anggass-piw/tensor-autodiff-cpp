#include "tensor.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace tinyad {
namespace {

std::size_t element_count(const std::vector<std::size_t>& shape) {
    std::size_t count = 1; // The empty shape represents a scalar.
    for (const auto dim : shape) {
        if (dim == 0) {
            throw std::invalid_argument("tensor dimensions must be positive");
        }
        if (count > std::numeric_limits<std::size_t>::max() / dim) {
            throw std::overflow_error("tensor shape element count overflows size_t");
        }
        count *= dim;
    }
    return count;
}

void require_same_shape(const Tensor& lhs, const Tensor& rhs, const char* operation) {
    if (lhs.shape() != rhs.shape()) {
        throw std::invalid_argument(std::string(operation) + " requires identical tensor shapes");
    }
}

void accumulate(Tensor& destination, const Tensor& contribution) {
    require_same_shape(destination, contribution, "gradient accumulation");
    for (std::size_t i = 0; i < destination.size(); ++i) {
        destination.flat_at(i) += contribution.values()[i];
    }
}

std::shared_ptr<Node> make_node(Tensor data,
                                bool requires_grad,
                                std::vector<std::shared_ptr<Node>> parents,
                                std::function<void()> backward_fn) {
    auto node = std::make_shared<Node>(std::move(data), requires_grad, false);
    node->parents = std::move(parents);
    node->backward_fn = std::move(backward_fn);
    return node;
}

} // namespace

Tensor::Tensor(std::vector<std::size_t> shape, std::vector<double> values)
    : shape_(std::move(shape)), values_(std::move(values)) {
    const auto expected = element_count(shape_);
    if (values_.size() != expected) {
        throw std::invalid_argument("tensor data size does not match its shape");
    }
}

Tensor::Tensor(std::vector<std::size_t> shape, double fill_value)
    : shape_(std::move(shape)), values_(element_count(shape_), fill_value) {}

Tensor Tensor::scalar(double value) {
    return Tensor({}, std::vector<double>{value});
}

double Tensor::item() const {
    if (size() != 1) {
        throw std::logic_error("item() requires a tensor containing exactly one element");
    }
    return values_[0];
}

double& Tensor::flat_at(std::size_t index) {
    if (index >= size()) throw std::out_of_range("flat tensor index is out of range");
    return values_[index];
}

const double& Tensor::flat_at(std::size_t index) const {
    if (index >= size()) throw std::out_of_range("flat tensor index is out of range");
    return values_[index];
}

double Tensor::at2d(std::size_t row, std::size_t col) const {
    if (shape_.size() != 2 || row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("2-D tensor index is out of range");
    }
    return values_[row * shape_[1] + col];
}

double& Tensor::at2d(std::size_t row, std::size_t col) {
    if (shape_.size() != 2 || row >= shape_[0] || col >= shape_[1]) {
        throw std::out_of_range("2-D tensor index is out of range");
    }
    return values_[row * shape_[1] + col];
}

Tensor Tensor::zeros_like() const {
    return Tensor(shape_, 0.0);
}

Node::Node(Tensor value, bool tracks_grad, bool leaf)
    : data(std::move(value)), grad(data.zeros_like()), requires_grad(tracks_grad), is_leaf(leaf) {}

Value Value::variable(Tensor data) {
    return Value(std::make_shared<Node>(std::move(data), true, true));
}

Value Value::constant(Tensor data) {
    return Value(std::make_shared<Node>(std::move(data), false, true));
}

const Tensor& Value::data() const noexcept { return node_->data; }

double& Value::mutable_data(std::size_t flat_index) {
    if (!node_->is_leaf) {
        throw std::logic_error("only leaf values may be changed in place");
    }
    return node_->data.flat_at(flat_index);
}

const Tensor& Value::grad() const noexcept { return node_->grad; }
bool Value::requires_grad() const noexcept { return node_->requires_grad; }

void Value::zero_grad() {
    for (std::size_t i = 0; i < node_->grad.size(); ++i) node_->grad.flat_at(i) = 0.0;
}

void Value::backward() {
    if (node_->data.size() != 1) {
        throw std::logic_error("backward() requires a scalar output; call sum() first");
    }

    std::vector<std::shared_ptr<Node>> topological;
    std::vector<std::pair<std::shared_ptr<Node>, bool>> dfs{{node_, false}};
    std::vector<Node*> visited;
    while (!dfs.empty()) {
        auto entry = std::move(dfs.back());
        dfs.pop_back();
        if (entry.second) {
            topological.push_back(std::move(entry.first));
            continue;
        }
        if (std::find(visited.begin(), visited.end(), entry.first.get()) != visited.end()) continue;
        visited.push_back(entry.first.get());
        dfs.emplace_back(entry.first, true);
        for (const auto& parent : entry.first->parents) dfs.emplace_back(parent, false);
    }

    // Keep leaf gradients cumulative, but reset transient graph-node gradients so a
    // second backward() propagates only the newly seeded output gradient.
    for (const auto& current : topological) {
        if (!current->is_leaf) {
            for (std::size_t i = 0; i < current->grad.size(); ++i) current->grad.flat_at(i) = 0.0;
        }
    }
    node_->grad.flat_at(0) += 1.0;
    for (auto it = topological.rbegin(); it != topological.rend(); ++it) {
        if ((*it)->backward_fn) (*it)->backward_fn();
    }
}

Value operator+(const Value& lhs, const Value& rhs) {
    require_same_shape(lhs.data(), rhs.data(), "addition");
    Tensor output(lhs.data().shape(), 0.0);
    for (std::size_t i = 0; i < output.size(); ++i) {
        output.flat_at(i) = lhs.data().values()[i] + rhs.data().values()[i];
    }
    auto a = lhs.node_;
    auto b = rhs.node_;
    const bool tracks = a->requires_grad || b->requires_grad;
    auto node = make_node(std::move(output), tracks, {a, b}, {});
    Node* out = node.get();
    node->backward_fn = [a, b, out]() {
        if (a->requires_grad) accumulate(a->grad, out->grad);
        if (b->requires_grad) accumulate(b->grad, out->grad);
    };
    return Value(std::move(node));
}

Value operator*(const Value& lhs, const Value& rhs) {
    require_same_shape(lhs.data(), rhs.data(), "elementwise multiplication");
    Tensor output(lhs.data().shape(), 0.0);
    for (std::size_t i = 0; i < output.size(); ++i) {
        output.flat_at(i) = lhs.data().values()[i] * rhs.data().values()[i];
    }
    auto a = lhs.node_;
    auto b = rhs.node_;
    auto node = make_node(std::move(output), a->requires_grad || b->requires_grad, {a, b}, {});
    Node* out = node.get();
    node->backward_fn = [a, b, out]() {
        if (a->requires_grad) {
            Tensor contribution(a->data.shape(), 0.0);
            for (std::size_t i = 0; i < contribution.size(); ++i)
                contribution.flat_at(i) = out->grad.values()[i] * b->data.values()[i];
            accumulate(a->grad, contribution);
        }
        if (b->requires_grad) {
            Tensor contribution(b->data.shape(), 0.0);
            for (std::size_t i = 0; i < contribution.size(); ++i)
                contribution.flat_at(i) = out->grad.values()[i] * a->data.values()[i];
            accumulate(b->grad, contribution);
        }
    };
    return Value(std::move(node));
}

Value operator+(const Value& lhs, double rhs) {
    Tensor output = lhs.data();
    for (std::size_t i = 0; i < output.size(); ++i) output.flat_at(i) += rhs;
    auto parent = lhs.node_;
    auto node = make_node(std::move(output), parent->requires_grad, {parent}, {});
    Node* out = node.get();
    node->backward_fn = [parent, out]() {
        if (parent->requires_grad) accumulate(parent->grad, out->grad);
    };
    return Value(std::move(node));
}

Value operator+(double lhs, const Value& rhs) { return rhs + lhs; }

Value operator*(const Value& lhs, double rhs) {
    Tensor output = lhs.data();
    for (std::size_t i = 0; i < output.size(); ++i) output.flat_at(i) *= rhs;
    auto parent = lhs.node_;
    auto node = make_node(std::move(output), parent->requires_grad, {parent}, {});
    Node* out = node.get();
    node->backward_fn = [parent, out, rhs]() {
        if (!parent->requires_grad) return;
        Tensor contribution = out->grad;
        for (std::size_t i = 0; i < contribution.size(); ++i) contribution.flat_at(i) *= rhs;
        accumulate(parent->grad, contribution);
    };
    return Value(std::move(node));
}

Value operator*(double lhs, const Value& rhs) { return rhs * lhs; }

Value Value::sum() const {
    double total = 0.0;
    for (const auto value : node_->data.values()) total += value;
    auto parent = node_;
    auto node = make_node(Tensor::scalar(total), parent->requires_grad, {parent}, {});
    Node* out = node.get();
    node->backward_fn = [parent, out]() {
        if (!parent->requires_grad) return;
        Tensor contribution(parent->data.shape(), out->grad.item());
        accumulate(parent->grad, contribution);
    };
    return Value(std::move(node));
}

Value Value::relu() const {
    Tensor output = node_->data;
    for (std::size_t i = 0; i < output.size(); ++i) {
        const double value = output.values()[i];
        output.flat_at(i) = value > 0.0 ? value : 0.0;
    }
    auto parent = node_;
    auto node = make_node(std::move(output), parent->requires_grad, {parent}, {});
    Node* out = node.get();
    node->backward_fn = [parent, out]() {
        if (!parent->requires_grad) return;
        Tensor contribution(parent->data.shape(), 0.0);
        for (std::size_t i = 0; i < contribution.size(); ++i)
            contribution.flat_at(i) = parent->data.values()[i] > 0.0 ? out->grad.values()[i] : 0.0;
        accumulate(parent->grad, contribution);
    };
    return Value(std::move(node));
}

Value Value::matmul(const Value& rhs) const {
    const auto& a_data = node_->data;
    const auto& b_data = rhs.node_->data;
    if (a_data.shape().size() != 2 || b_data.shape().size() != 2) {
        throw std::invalid_argument("matmul supports only 2-D tensors");
    }
    const std::size_t rows = a_data.shape()[0];
    const std::size_t inner = a_data.shape()[1];
    const std::size_t rhs_inner = b_data.shape()[0];
    const std::size_t cols = b_data.shape()[1];
    if (inner != rhs_inner) throw std::invalid_argument("matmul inner dimensions must match");

    Tensor output({rows, cols}, 0.0);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t k = 0; k < inner; ++k) {
            const double left = a_data.at2d(i, k);
            for (std::size_t j = 0; j < cols; ++j)
                output.at2d(i, j) += left * b_data.at2d(k, j);
        }
    }

    auto a = node_;
    auto b = rhs.node_;
    auto node = make_node(std::move(output), a->requires_grad || b->requires_grad, {a, b}, {});
    Node* out = node.get();
    node->backward_fn = [a, b, out, rows, inner, cols]() {
        if (a->requires_grad) {
            Tensor contribution(a->data.shape(), 0.0);
            for (std::size_t i = 0; i < rows; ++i)
                for (std::size_t k = 0; k < inner; ++k)
                    for (std::size_t j = 0; j < cols; ++j)
                        contribution.at2d(i, k) += out->grad.at2d(i, j) * b->data.at2d(k, j);
            accumulate(a->grad, contribution);
        }
        if (b->requires_grad) {
            Tensor contribution(b->data.shape(), 0.0);
            for (std::size_t k = 0; k < inner; ++k)
                for (std::size_t j = 0; j < cols; ++j)
                    for (std::size_t i = 0; i < rows; ++i)
                        contribution.at2d(k, j) += a->data.at2d(i, k) * out->grad.at2d(i, j);
            accumulate(b->grad, contribution);
        }
    };
    return Value(std::move(node));
}

} // namespace tinyad
