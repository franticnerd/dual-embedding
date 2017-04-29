#include "base.h"
#include "utility.h"

#include <vector>
    
#define EPOCHS 100

class LabelPropagation : public Model {
    int size_, dim_;
    std::vector<std::vector<double>> embedding_;
    void UpdateEmbedding(const Graph& base, int x);
  public:
    LabelPropagation(const Graph& base, const Label& label);
    const std::vector<double>& GetEmbedding(int x) { return embedding_[x]; }
};

void LabelPropagation::UpdateEmbedding(const Graph& base, int x) {
    fill(embedding_[x].begin(), embedding_[x].end(), 0);
    for (int y : base.edge[x])
        for (int i = 0; i < dim_; ++i)
            embedding_[x][i] += embedding_[y][i] / base.edge[x].size();
}

LabelPropagation::LabelPropagation(const Graph& base, const Label& label) : 
    size_(base.size),
    dim_(label.card),
    embedding_(base.size) {

    for (int i = 0; i < size_; ++i) {
        embedding_[i].resize(dim_, 0);
        if (label.label[i] != -1)
            embedding_[i][label.label[i]] = 1;
    }

    std::vector<int> order(size_);
    for (int j = 0; j < size_; ++j)
        order[j] = j;
    for (int i = 0; i < EPOCHS; ++i) {
        RandomPermutation(&order);
        for (int j : order)
            if (label.label[j] == -1)
                UpdateEmbedding(base, j);
    }
}

Model* GetLabelPropagation(const Graph& base, const Label& label) {
    return new LabelPropagation(base, label);
}