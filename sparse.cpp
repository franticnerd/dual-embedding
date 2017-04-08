#include "base.h"
#include "utility.h"
#include "svm.h"
#include <vector>
#include <map>

#define EPOCHS 10

struct SparseFeature {
    int index;
    double value;
    SparseFeature(int index_, double value_) : index(index_), value(value_) {}
};

class SparseEmbedding : public Model {
    int size_;
    std::vector<std::vector<SparseFeature>> embedding;
    std::vector<std::vector<double>> coeff;

    void UpdateEmbedding(const Graph& positive, const Graph& negative, int x);
public:
    SparseEmbedding(const Graph& graph);
    double Evaluate(int x, int y);
};

void SparseEmbedding::UpdateEmbedding(const Graph& positive, const Graph& negative, int x) {
    std::map<int, int> feature_index;
    for (int i = 0; i < (int)embedding[x].size(); ++i)
        feature_index[embedding[x][i].index] = i;

    std::vector<int> label, instance;
    for (int i : positive.edge[x]) {
        instance.push_back(i);
        label.push_back(1);
    }
    for (int i : negative.edge[x]) {
        instance.push_back(i);
        label.push_back(-1);
    }

    std::vector<std::vector<double>> feature;
    std::vector<std::vector<double>*> feature_ptr;
    for (int i : instance) {
        std::vector<double> vec(embedding[x].size());
        for (const auto& p : embedding[i])
            if (feature_index.count(p.index) > 0)
                vec[feature_index[p.index]] = p.value;
        feature.push_back(vec);
        feature_ptr.push_back(&feature.back());
    }

    LinearSVM(feature_ptr, label, 1, 1, &coeff[x]);
    std::vector<double> val(embedding[x].size(), 0);
    for (int i = 0; i < (int)instance.size(); ++i) {        
        for (int j = 0; j < (int)embedding[x].size(); ++j)
            val[j] += coeff[x][i] * feature[i][j];
    }
    for (int i = 0; i < (int)embedding[x].size(); ++i)
        embedding[x][i].value = val[i];
}

SparseEmbedding::SparseEmbedding(const Graph& graph) :
    size_(graph.size) {
    embedding.resize(size_);
    for (int i = 0; i < size_; ++i) {
        embedding[i].clear();
        embedding[i].push_back(SparseFeature(i, 1));
        for (int j : graph.edge[i])
            embedding[i].push_back(SparseFeature(j, 0));
    }

    Graph negative(size_);
    SampleLocalNegativeGraph(graph, &negative);

    coeff.resize(size_);
    for (int i = 0; i < size_; ++i)
        coeff[i].resize(graph.edge[i].size() + negative.edge[i].size());

    std::vector<int> order(size_);
    for (int j = 0; j < size_; ++j)
        order[j] = j;
    for (int i = 0; i < EPOCHS; ++i) {
        RandomPermutation(&order);
        for (int j : order)
            UpdateEmbedding(graph, negative, j);
    }
}

double SparseEmbedding::Evaluate(int x, int y) {
    std::map<int, double> map;
    for (const auto& p : embedding[x])
        map[p.index] = p.value;
    double val = 0;
    for (const auto& p : embedding[y])
        if (map.count(p.index) > 0)
            val += p.value * map[p.index];
    return val;
}

Model* GetSparseEmbedding(const Graph& graph) {
    return new SparseEmbedding(graph);
}