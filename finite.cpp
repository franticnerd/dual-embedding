#include "base.h"
#include "utility.h"
#include "svm.h"
#include <vector>
#include <random>

namespace {
    std::mt19937 gen;
}   // anonymous namespace

#define EPOCS 10

class FiniteEmbedding : public Model {
    int size_, dim_;
    std::vector<std::vector<double>> embedding;

    void UpdateEmbedding(const Graph& positive, const Graph& negative, int x);
  public:
    FiniteEmbedding(const Graph& graph, int dimension);
    double Evaluate(int x, int y);
};

void FiniteEmbedding::UpdateEmbedding(const Graph& positive, const Graph& negative, int x) {
    std::vector<std::vector<double>*> feature;
    std::vector<int> label;
    for (int i : positive.edge[x]) {
        feature.push_back(&embedding[i]);
        label.push_back(1);
    }
    for (int i : negative.edge[x]) {
        feature.push_back(&embedding[i]);
        label.push_back(-1);
    }
    std::vector<double> coeff;
    LinearSVM(feature, label, 1, 1, &coeff);
    for (int i = 0; i < dim_; ++i) {
        double val = 0;
        for (int j = 0; j < (int)feature.size(); ++j)
            val += coeff[j] * feature[j]->at(i);
        embedding[x][i] = val;
    }
}

FiniteEmbedding::FiniteEmbedding(const Graph& graph, int dimension) :
    size_(graph.size),
    dim_(dimension) {
    std::uniform_real_distribution<double> dist(-1, 1);
    embedding.resize(graph.size);
    for (int i = 0; i < graph.size; ++i)
        embedding[i].resize(dimension);
    for (int i = 0; i < graph.size; ++i)
        for (int j = 0; j < dimension; ++j)
            embedding[i][j] = dist(gen);
    Graph negative(graph.size);
    SampleNegativeGraph(graph, &negative);

    std::vector<int> order(graph.size);
    for (int j = 0; j < graph.size; ++j)
        order[j] = j;
    for (int i = 0; i < EPOCS; ++i) {
        RandomPermutation(&order);
        for (int j = 0; j < graph.size; ++j)
            UpdateEmbedding(graph, negative, order[j]);
    }
}

double FiniteEmbedding::Evaluate(int x, int y) {
    return InnerProduct(embedding[x], embedding[y]);
}

Model* GetFiniteEmbedding(const Graph& graph, int dimension) {
    return new FiniteEmbedding(graph, dimension);
}