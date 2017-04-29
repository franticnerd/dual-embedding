#include "base.h"
#include "utility.h"

#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>
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

class Predefined : public Model {
    int n_, dim_;
    std::vector<std::vector<double>> embedding;
  public:
    Predefined(const std::string& filename) {
        char buffer[1200];

        std::ifstream fin(filename);
        fin.getline(buffer, 1200);
        std::istringstream is(buffer);
        is >> n_ >> dim_;

        embedding.resize(n_);
        for (int i = 0; i < n_; ++i) {
            embedding[i].resize(dim_);
            fin.getline(buffer, 1200);
            
            std::istringstream is2(buffer);
            std::string word;
            std::vector<std::string> word_vec;
            while (is2 >> word)
                word_vec.push_back(word);
            for (int j = word_vec.size() - dim_; j < (int)word_vec.size(); ++j)
                embedding[i].push_back(std::stof(word_vec[j]));
        }
    }
    double Evaluate(int x, int y) {
        return InnerProduct(embedding[x].data(), embedding[y].data(), dim_);
    }
};

double EvaluateAveragePrecision(Model* model, const Graph& pos, const Graph& neg) {
    std::vector<double> p, n;
    for (int i = 0; i < pos.size; ++i)
        for (int y : pos.edge[i])
            p.push_back(model->Evaluate(i, y));
    for (int i = 0; i < neg.size; ++i)
        for (int y : neg.edge[i])
            n.push_back(model->Evaluate(i, y));
    return EvaluateAveragePrecision(p, n);
}

void EvaluateAll(Model* model, const Graph& train_pos, const Graph& train_neg, const Graph& test_pos, const Graph& test_neg) {
    double v, cnt;
    std::cout << "Test Average Precision\n" << EvaluateAveragePrecision(model, test_pos, test_neg) << "\n";
    std::cout << "Empirical Average Precision\n" << EvaluateAveragePrecision(model, train_pos, train_neg) << "\n";
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

void EvalFiniteEmbedding(const Graph& train, const Graph& neg_train, const Graph& test, const Graph& neg_test) {
    std::unique_ptr<Model> model;
    std::cout << "Training Finite Embedding\n";
    model.reset(GetFiniteEmbedding(train, neg_train, 100, 0.03, 30, 0));
    std::cout << "Evaluating Finite Embedding\n";
    EvaluateAll(model.get(), train, neg_train, test, neg_test);
}

void EvalFiniteContrastEmbedding(const Graph& train, const Graph& neg_train, const Graph& test, const Graph& neg_test) {
    std::unique_ptr<Model> model;
    std::cout << "Training Finite Contrast Embedding\n";
    model.reset(GetFiniteContrastEmbedding(train, neg_train, 6, 100, 100, 0));
    std::cout << "Evaluating Finite Contrast Embedding\n";
    EvaluateAll(model.get(), train, neg_train, test, neg_test);
}

void EvalKernelEmbedding(const Graph& train, const Graph& neg_train, const Graph& test, const Graph& neg_test) {
    std::unique_ptr<Model> model;
    std::cout << "Training Kernel Embedding\n";
    model.reset(GetKernelEmbedding(train, neg_train, 0.03, 30, 0));
    std::cout << "Evaluating Kernel Embedding\n";
    EvaluateAll(model.get(), train, neg_train, test, neg_test);
}

void EvalSparseEmbedding(const Graph& train, const Graph& neg_train, const Graph& test, const Graph& neg_test) {
    std::unique_ptr<Model> model;
    std::cout << "Training Sparse Embedding\n";
    model.reset(GetSparseEmbedding(train, neg_train, 0.015, 15, 0));
    std::cout << "Evaluating Sparse Embedding\n";
    EvaluateAll(model.get(), train, neg_train, test, neg_test);
}

void EvalSequentialEmbedding(const Graph& train, const Graph& neg_train, const Graph& test, const Graph& neg_test) {
    std::unique_ptr<Model> model;
    std::cout << "Training Sequential Embedding\n";
    model.reset(GetSequentialFiniteEmbedding(train, neg_train, 100, 0.1, 2));
    std::cout << "Evaluating Sequential Embedding\n";
    EvaluateAll(model.get(), train, neg_train, test, neg_test);
}

void EvalCommonNeighbor(const Graph& train, const Graph& neg_train, const Graph& test, const Graph& neg_test) {
    std::unique_ptr<Model> model;
    std::cout << "Training Common Neighbor\n";
    model.reset(GetCommonNeighbor(train, 120));
    std::cout << "Evaluating Common Neighbor\n";
    EvaluateAll(model.get(), train, neg_train, test, neg_test);
}

void EvalPredefined(const Graph& train, const Graph& neg_train, const Graph& test, const Graph& neg_test) {
    std::unique_ptr<Model> model;
    std::cout << "Training Predefined\n";
    model.reset(new Predefined("vec-filter.txt"));
    std::cout << "Evaluating Predefined\n";
    std::cout << EvaluateAveragePrecision(model.get(), test, neg_test) << "\n";
}

void EvalRandom(const Graph& test, const Graph& neg_test) {
    std::unique_ptr<Model> model(new Random());
    std::cout << "Evaluating Random\n";
    std::cout << EvaluateAveragePrecision(model.get(), test, neg_test) << "\n";
}

int main() {
    Graph train, test;
    std::cout << "Reading Dataset\n";
    ReadDataset("node-w-filter.txt", "edge-ww-train-filter.txt", &train);
    ReadDataset("node-w-filter.txt", "edge-ww-val-filter.txt", &test);

    std::cout << "Sampling Negative Dataset\n";
    Graph neg_train(train.size), neg_test(test.size), neg_empty(train.size);
    
    SampleNegativeGraphUniform(train, &neg_train);
    //SampleNegativeGraphPreferential(test, &neg_test, 1);    
    SampleNegativeGraphUniform(test, &neg_test);
    //SampleNegativeGraphLocal(train, &neg_local);
    RemoveRedundant(train, &neg_test);
    RemoveRedundant(test, &neg_test);

    EvalFiniteEmbedding(train, neg_train, test, neg_test);
    EvalFiniteContrastEmbedding(train, neg_train, test, neg_test);
    //EvalKernelEmbedding(train, neg_train, test, neg_test);
    //EvalSparseEmbedding(train, neg_empty, test, neg_test);
    //EvalSequentialEmbedding(train, neg_train, test, neg_test);
    EvalCommonNeighbor(train, neg_train, test, neg_test);
    EvalPredefined(train, neg_train, test, neg_test);
    EvalRandom(test, neg_test);

    system("pause");
}