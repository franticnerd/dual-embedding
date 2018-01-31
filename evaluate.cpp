#include "base.h"
#include "utility.h"
#include "svm.h"

#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <memory>
#include <unordered_set>

namespace {
    std::mt19937 gen(9119);
}   // anonymous namespace

#define EPOCHS 100
#define LINK_EPOCHS 20

double EvaluatePredictedAP(Model* model, const Graph& train, const Graph& pos, const Graph& neg, double regularizer, int sample_ratio) {
    int dim = model->GetEmbedding(0).size();
    std::uniform_int_distribution<int> dist(0, train.size - 1);

    std::vector<std::vector<double>> vec;
    std::vector<double> norm, penalty_coeff, margin;
    std::vector<int> label;
    for (int x = 0; x < train.size; ++x)
        for (int y : train.edge[x]) {
            std::vector<double> edge_vec(dim);
            for (int j = 0; j < dim; ++j)
                edge_vec[j] = model->GetEmbedding(x)[j] * model->GetEmbedding(y)[j];
            for (int i = 0; i < sample_ratio; ++i) {
                std::vector<double> contrast(dim);
                int xp = dist(gen), yp = dist(gen);
                for (int j = 0; j < dim; ++j)
                    contrast[j] = edge_vec[j] - model->GetEmbedding(xp)[j] * model->GetEmbedding(yp)[j];
                norm.push_back(InnerProduct(contrast.data(), contrast.data(), dim));
                label.push_back(1);
                penalty_coeff.push_back(1 / regularizer);
                margin.push_back(1);
                vec.push_back(std::move(contrast));
            }           
        }

    std::vector<double> coeff(vec.size(), 0), w(dim, 0);
    std::vector<const double*> ptr_vec;
    for (int i = 0; i < (int)vec.size(); ++i)
        ptr_vec.push_back(vec[i].data());

    for (int i = 0; i < LINK_EPOCHS; ++i)
        LinearSVM(ptr_vec, norm, label, penalty_coeff, margin, &coeff, &w, dim, false);

    std::vector<double> p, n;
    for (int x = 0; x < train.size; ++x)
        for (int y : pos.edge[x]) {
            std::vector<double> edge_vec(dim);
            for (int j = 0; j < dim; ++j)
                edge_vec[j] = model->GetEmbedding(x)[j] * model->GetEmbedding(y)[j];
            p.push_back(InnerProduct(edge_vec.data(), w.data(), dim));
        }

    for (int x = 0; x < train.size; ++x)
        for (int y : neg.edge[x]) {
            std::vector<double> edge_vec(dim);
            for (int j = 0; j < dim; ++j)
                edge_vec[j] = model->GetEmbedding(x)[j] * model->GetEmbedding(y)[j];
            n.push_back(InnerProduct(edge_vec.data(), w.data(), dim));
        }
    return EvaluateAveragePrecision(p, n);
}

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

double EvaluateAveragePrecision(Model* model, const DGraph& pos, const DGraph& neg) {
    std::vector<double> p, n;
    for (int i = 0; i < pos.size; ++i)
        for (int y : pos.out_edge[i])
            p.push_back(model->Evaluate(i, y));
    for (int i = 0; i < neg.size; ++i)
        for (int y : neg.out_edge[i])
            n.push_back(model->Evaluate(i, y));
    return EvaluateAveragePrecision(p, n);
}

double EvaluateF1(Model* model, const Label& train, const Label& test, double regularizer, int sample_ratio, bool normalize) {
    int dim = model->GetEmbedding(0).size();
    std::vector<const double*> vec;
    for (int i = 0; i < train.size; ++i)
        vec.push_back(model->GetEmbedding(i).data());
    std::vector<double> v_norm(train.size, 0);
    for (int i = 0; i < train.size; ++i)
        if (normalize)
            v_norm[i] = sqrt(InnerProduct(vec[i], vec[i], dim));
        else
            v_norm[i] = 1;

    double ave_f1 = 0;
    for (int a = 0; a < train.card; ++a) {
        std::unordered_set<int> positive(test.label_instance[a].begin(), test.label_instance[a].end());
        std::uniform_int_distribution<int> dist(0, train.size - 1);

        std::vector<std::vector<double>> train_vec;
        std::vector<double> norm, penalty_coeff, margin;
        std::vector<int> label;
        for (int i : train.label_instance[a]) {
            for (int j = 0; j < sample_ratio; ++j) {
                int t = dist(gen);
                while (!train.labeled[t] || positive.count(t) > 0)
                    t = dist(gen);
                std::vector<double> feature_vec(dim);
                for (int k = 0; k < dim; ++k) 
                    feature_vec[k] = vec[i][k] / std::max(v_norm[i], 1e-4) - vec[t][k] / std::max(v_norm[t], 1e-4);

                norm.push_back(InnerProduct(feature_vec.data(), feature_vec.data(), dim));
                label.push_back(1);
                penalty_coeff.push_back(1 / regularizer);
                margin.push_back(1);
                train_vec.push_back(std::move(feature_vec));
            }
        }

        std::vector<const double*> ptr_vec;
        for (int i = 0; i < (int)train_vec.size(); ++i)
            ptr_vec.push_back(train_vec[i].data());

        std::vector<double> coeff(train_vec.size(), 0), w(dim, 0), label_prediction(train.size);
        for (int i = 0; i < EPOCHS; ++i)
            LinearSVM(ptr_vec, norm, label, penalty_coeff, margin, &coeff, &w, dim, false);

        for (int i = 0; i < train.size; ++i)
            label_prediction[i] = InnerProduct(vec[i], w.data(), dim) / v_norm[i];

        std::vector<double> p, n;
        for (int i = 0; i < test.size; ++i)
            if (positive.count(i) > 0)
                p.push_back(label_prediction[i]);
            else if (test.labeled[i])
                n.push_back(label_prediction[i]);

        ave_f1 += EvaluateF1(p, n);
        std::cout << "Processing " << a << "; Ave: " << ave_f1 / (a + 1) << "\n";
    }

    ave_f1 /= train.card;
    return ave_f1;
}

double EvaluateF1LabelPropagation(const Graph& base, const Label& train, const Label& test) {
    double ave_f1 = 0;
    int total = 0;
    for (int a = 0; a < test.card; ++a) {
        SingleLabel label(train.size);
        for (int i = 0; i < train.size; ++i)
            if (train.labeled[i])
                label.SetLabel(i, 0);
        for (int i : train.label_instance[a])
            label.SetLabel(i, 1);

        std::unique_ptr<Model> model(GetLabelPropagation(base, label));
        std::vector<double> p, n;
        std::unordered_set<int> pos(test.label_instance[a].begin(), test.label_instance[a].end());
        for (int i = 0; i < test.size; ++i)
            if (pos.count(i) > 0)
                p.push_back(model->GetEmbedding(i)[1]);
            else if (test.labeled[i])
                n.push_back(model->GetEmbedding(i)[1]);

        if (p.size() > 0)
            total++;
        else
            continue;
        ave_f1 += EvaluateF1(p, n);
        std::cout << "Processing " << total << "; Ave: " << ave_f1 / total << "\n";
    }
    ave_f1 /= test.card;
    return ave_f1;
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
            cnt++;
        }
    std::cout << v << " / " << cnt << "\n";
    std::cout << "Total L2 Norm\n";
    v = 0;
    for (int i = 0; i < train_pos.size; ++i)
        v += model->Evaluate(i, i);
    std::cout << v << "\n";
}

void EvaluateAll(Model* model, const DGraph& train_pos, const DGraph& train_neg, const DGraph& test_pos, const DGraph& test_neg) {
    double v, cnt;
    std::cout << "Test Average Precision\n" << EvaluateAveragePrecision(model, test_pos, test_neg) << "\n";
    std::cout << "Empirical Average Precision\n" << EvaluateAveragePrecision(model, train_pos, train_neg) << "\n";
    std::cout << "Empirical Hinge Loss(Positive)\n";
    v = cnt = 0;
    for (int i = 0; i < train_pos.size; ++i)
        for (int y : train_pos.out_edge[i]) {
            v += std::max(1 - model->Evaluate(i, y), (double)0);
            cnt++;
        }
    std::cout << v << " / " << cnt << "\n";
    std::cout << "Empirical Hinge Loss(Negative)\n";
    v = cnt = 0;
    for (int i = 0; i < train_neg.size; ++i)
        for (int y : train_neg.out_edge[i]) {
            v += std::max(model->Evaluate(i, y), (double)0);
            cnt++;
        }
    std::cout << v << " / " << cnt << "\n";
    std::cout << "Test Hinge Loss(Positive)\n";
    v = cnt = 0;
    for (int i = 0; i < test_pos.size; ++i)
        for (int y : test_pos.out_edge[i]) {
            v += std::max(1 - model->Evaluate(i, y), (double)0);
            cnt++;
        }
    std::cout << v << " / " << cnt << "\n";
    std::cout << "Test Hinge Loss(Negative)\n";
    v = cnt = 0;
    for (int i = 0; i < test_neg.size; ++i)
        for (int y : test_neg.out_edge[i]) {
            v += std::max(model->Evaluate(i, y), (double)0);
            cnt++;
        }
    std::cout << v << " / " << cnt << "\n";
    std::cout << "Total L2 Norm\n";
    v = 0;
    for (int i = 0; i < train_pos.size; ++i)
        v += InnerProduct(model->GetEmbedding(i).data(), model->GetEmbedding(i).data(), model->GetEmbedding(i).size());
    std::cout << v << "\n";
}
