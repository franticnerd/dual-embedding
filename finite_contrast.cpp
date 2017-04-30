#include "base.h"
#include "utility.h"
#include "svm.h"
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>

namespace {
    std::mt19937 gen;
}   // anonymous namespace

#define EPOCHS 10

struct ContrastEdgePair {
    int b, c, d, label;
    ContrastEdgePair(int b_, int c_, int d_, int label_) :
        b(b_), c(c_), d(d_), label(label_) {}
};

typedef std::vector<std::vector<ContrastEdgePair>> ContrastEdgeAdjacencyList;


class FiniteContrastEmbedding : public Model {
    int size_, dim_;
    const double regularizer_, deg_norm_pow_;
    std::vector<std::vector<double>> embedding;
    std::vector<std::vector<double>> coeff;
    std::vector<double> sqr_norm;

    void UpdateEmbedding(const ContrastEdgeAdjacencyList& table, int x);
public:
    FiniteContrastEmbedding(const Graph& graph, const Graph& negative, int sample_ratio, int dimension, double regularizer, double deg_norm_pow);
    double Evaluate(int x, int y);
};

void FiniteContrastEmbedding::UpdateEmbedding(const ContrastEdgeAdjacencyList& table, int x) {
    double deg_norm = pow(std::max((int)table[x].size(), 1), deg_norm_pow_);

    std::vector<const double*> feature;
    std::vector<int> label;
    std::vector<double> margin, penalty_coeff, f_sqr_norm;
    for (const ContrastEdgePair& pair : table[x]) {
        feature.push_back(embedding[pair.b].data());
        label.push_back(pair.label);
        margin.push_back(1 + pair.label * InnerProduct(embedding[pair.c].data(), embedding[pair.d].data(), dim_));
        penalty_coeff.push_back(deg_norm / regularizer_);
        f_sqr_norm.push_back(sqr_norm[pair.b]);
    }
    LinearSVM(feature, f_sqr_norm, label, penalty_coeff, margin, &coeff[x], &embedding[x], dim_, false);
    sqr_norm[x] = InnerProduct(embedding[x].data(), embedding[x].data(), dim_);
}

FiniteContrastEmbedding::FiniteContrastEmbedding(const Graph& graph, const Graph& negative, int sample_ratio, int dimension, double regularizer, double deg_norm_pow) :
    size_(graph.size),
    dim_(dimension),
    regularizer_(regularizer),
    deg_norm_pow_(deg_norm_pow) {

    std::uniform_real_distribution<double> dist_d(-1, 1);
    embedding.resize(size_);
    for (int i = 0; i < size_; ++i)
        embedding[i].resize(dim_);
    for (int i = 0; i < size_; ++i)
        for (int j = 0; j < dim_; ++j)
            embedding[i][j] = dist_d(gen);

    // Construct Contrast Pair Adjacency List
    ContrastEdgeAdjacencyList table(size_);
    std::vector<std::pair<int, int>> edge_list;
    for (int x = 0; x < size_; ++x)
        for (int y : negative.edge[x])
            edge_list.push_back(std::make_pair(x, y));
    std::uniform_int_distribution<int> dist_i(0, edge_list.size() - 1);
    for (int a = 0; a < size_; ++a)
        for (int b : graph.edge[a]) {
            int cnt = 0;
            while (1) {
                int i = dist_i(gen);
                int c = edge_list[i].first, d = edge_list[i].second;
                if (a == c || a == d || b == c || b == d) continue;
                table[a].push_back(ContrastEdgePair(b, c, d, 1));
                table[b].push_back(ContrastEdgePair(a, c, d, 1));
                table[c].push_back(ContrastEdgePair(d, a, b, -1));
                table[d].push_back(ContrastEdgePair(c, a, b, -1));
                if (++cnt == sample_ratio)
                    break;
            }
        }

    sqr_norm.resize(size_);
    for (int i = 0; i < size_; ++i)
        sqr_norm[i] = InnerProduct(embedding[i].data(), embedding[i].data(), dim_);

    coeff.resize(size_);
    for (int i = 0; i < size_; ++i)
        coeff[i].resize(table[i].size());

    std::vector<int> order(size_);
    for (int j = 0; j < size_; ++j)
        order[j] = j;
    for (int i = 0; i < EPOCHS; ++i) {
        RandomPermutation(&order);
        for (int j : order)
            UpdateEmbedding(table, j);
    }
}

double FiniteContrastEmbedding::Evaluate(int x, int y) {
    return InnerProduct(embedding[x].data(), embedding[y].data(), dim_);
}

Model* GetFiniteContrastEmbedding(const Graph& graph, const Graph& negative, int sample_ratio, int dimension, double regularizer, double deg_norm_pow) {
    return new FiniteContrastEmbedding(graph, negative, sample_ratio, dimension, regularizer, deg_norm_pow);
}