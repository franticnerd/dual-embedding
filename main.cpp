#include "base.h"
#include "utility.h"

#include <memory>
#include <iostream>
#include <random>
#include <algorithm>

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

void EvaluateMAP(Model* model, const Graph& pos, const Graph& neg) {
    std::vector<double> p, n;
    for (int i = 0; i < pos.size; ++i)
        for (int y : pos.edge[i])
            p.push_back(model->Evaluate(i, y));
    for (int i = 0; i < neg.size; ++i)
        for (int y : neg.edge[i])
            n.push_back(model->Evaluate(i, y));
    std::cout << EvaluateMAP(p, n) << "\n";
}

void EvaluateAll(Model* model, const Graph& train_pos, const Graph& train_neg, const Graph& test_pos, const Graph& test_neg) {
    double v, cnt;
    EvaluateMAP(model, test_pos, test_neg);
    std::cout << "Empirical MAP\n";
    EvaluateMAP(model, train_pos, train_neg);
    std::cout << "Empirical Hinge Loss(Positive)\n";
    v = cnt = 0;
    for (int i = 0; i < train_pos.size; ++i)
        for (int y : train_pos.edge[i]) {
            v += std::max(1 - model->Evaluate(i, y), (double)0);
            cnt++;
        }
    std::cout << v << " / " << cnt << "\n";
    std::cout << "Empirical Hinge Loss(Negative)\n";
    v = cnt = 0;
    for (int i = 0; i < train_neg.size; ++i)
        for (int y : train_neg.edge[i]) {
            v += std::max(model->Evaluate(i, y), (double)0);
            cnt++;
        }
    std::cout << v << " / " << cnt << "\n";
    std::cout << "Test Hinge Loss(Positive)\n";
    v = cnt = 0;
    for (int i = 0; i < test_pos.size; ++i)
        for (int y : test_pos.edge[i]) {
            v += std::max(1 - model->Evaluate(i, y), (double)0);
            cnt++;
        }
    std::cout << v << " / " << cnt << "\n";
    std::cout << "Test Hinge Loss(Negative)\n";
    v = cnt = 0;
    for (int i = 0; i < test_neg.size; ++i)
        for (int y : test_neg.edge[i]) {
            v += std::max(model->Evaluate(i, y), (double)0);
            cnt ++;
        }
    std::cout << v << " / " << cnt << "\n";
    std::cout << "Total L2 Norm\n";
    v = 0;
    for (int i = 0; i < train_pos.size; ++i)
        v += model->Evaluate(i, i);
    std::cout << v << "\n";
}

int main() {
    Graph train(0), test(0);
    std::cout << "Reading Dataset\n";
    ReadDataset("node-w.txt", "edge-ww.txt", &train, &test);
    Graph neg_train(train.size), neg_test(test.size), neg_empty(train.size), neg_local(train.size);

    std::cout << "Sampling Negative Dataset\n";
    SampleNegativeGraphPreferential(train, &neg_train, 0.1);
    SampleNegativeGraphLocal(train, &neg_local);
    SampleNegativeGraphUniform(test, &neg_test);
    //SampleNegativeGraphPreferential(test, &neg_test, 1);

    std::cout << "Training Finite Embedding\n";
    std::unique_ptr<Model> model(GetFiniteEmbedding(train, neg_train, 100));
    std::cout << "Evaluating Finite Embedding\n";
    EvaluateAll(model.get(), train, neg_train, test, neg_test);

    //std::cout << "Training Kernel Embedding\n";
    //model.reset(GetKernelEmbedding(train, neg_train, false, false));
    //std::cout << "Evaluating Kernel Embedding\n";
    //EvaluateAll(model.get(), train, neg_train, test, neg_test);

    std::cout << "Training Sparse Embedding\n";
    model.reset(GetSparseEmbedding(train, neg_empty));
    std::cout << "Evaluating Sparse Embedding\n";
    EvaluateAll(model.get(), train, neg_train, test, neg_test);

    std::cout << "Training Common Neighbor\n";
    model.reset(GetCommonNeighbor(train));
    std::cout << "Evaluating Common Neighbor\n";
    EvaluateMAP(model.get(), test, neg_test);

    model.reset(new Random());
    std::cout << "Evaluating Random\n";
    EvaluateMAP(model.get(), test, neg_test);

    system("pause");
}