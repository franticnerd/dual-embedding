#include "base.h"
#include "utility.h"

#include <memory>
#include <iostream>
#include <random>

namespace {
    std::mt19937 gen;
}   // anonymous namespace

class Random : public Model {
  public:
    Random() {}
    double Evaluate(int x, int y) {
        std::uniform_real_distribution<double> dist(0, 1);
        return dist(gen);
    }
};

void Evaluate(Model* model, const Graph& pos, const Graph& neg) {
    std::vector<double> p, n;
    for (int i = 0; i < pos.size; ++i)
        for (int y : pos.edge[i])
            p.push_back(model->Evaluate(i, y));
    for (int i = 0; i < neg.size; ++i)
        for (int y : neg.edge[i])
            n.push_back(model->Evaluate(i, y));
    std::cout << EvaluateMAP(p, n) << "\n";
}

int main() {
    Graph train(0), test(0);
    std::cout << "Reading Dataset\n";
    ReadDataset("node-w.txt", "edge-ww.txt", &train, &test);
    Graph neg_train(train.size), neg_test(test.size), neg_empty(train.size);
    std::cout << "Sampling Negative Dataset\n";
    SampleNegativeGraph(train, &neg_train);
    SampleNegativeGraph(test, &neg_test);
    std::cout << "Training Finite Embedding\n";
    std::unique_ptr<Model> m1(GetFiniteEmbedding(train, neg_train, 100));
    std::cout << "Training Kernel Embedding\n";
    std::unique_ptr<Model> m2(GetKernelEmbedding(test, neg_test));
    std::cout << "Training Sparse Embedding\n";
    std::unique_ptr<Model> m3(GetSparseEmbedding(train, neg_empty));
    std::cout << "Training Common Neighbor\n";
    std::unique_ptr<Model> m4(GetCommonNeighbor(train));
    
    std::unique_ptr<Model> m5(new Random());

    // Evaluate m1
    std::cout << "Finite Embedding\n";
    Evaluate(m1.get(), test, neg_test);

    std::cout << "Kernel Embedding\n";
    Evaluate(m2.get(), test, neg_test);

    std::cout << "Sparse Embedding\n";
    Evaluate(m3.get(), test, neg_test);

    std::cout << "Common Neighbor\n";
    Evaluate(m4.get(), test, neg_test);

    std::cout << "Random\n";
    Evaluate(m5.get(), test, neg_test);

    system("pause");
}