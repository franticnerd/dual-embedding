#include "base.h"
#include "utility.h"
#include "svm.h"

#include <iostream>
#include <vector>
#include <algorithm>

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

double EvaluateF1(Model* model, const Label& train, const Label& test, double regularizer) {
    std::vector<const double*> vec;
    for (int i = 0; i < train.size; ++i)
        vec.push_back(model->GetEmbedding(i).data());
    std::vector<std::vector<int>> label_prediction(train.size);
    for (int i = 0; i < train.size; ++i)
        label_prediction[i].resize(train.card, 0);
    for (int a = 0; a < train.card; ++a)
        for (int b = a + 1; b < train.card; ++b) {

            std::vector<const double*> train_vec;
            std::vector<double> norm, penalty_coeff, margin;
            std::vector<int> label;
            int dim = model->GetEmbedding(0).size();

            for (int i = 0; i < train.size; ++i) {
                if (train.label[i] == a || train.label[i] == b) {
                    train_vec.push_back(vec[i]);
                    norm.push_back(InnerProduct(vec[i], vec[i], dim));
                    label.push_back((train.label[i] == a ? 1 : -1));
                    penalty_coeff.push_back(1 / regularizer);
                    margin.push_back(1);
                }
            }

            std::vector<double> coeff(train_vec.size(), 0);
            std::vector<double> w(dim, 0);
            LinearSVM(train_vec, norm, label, penalty_coeff, margin, &coeff, &w, dim, false);

            for (int i = 0; i < train.size; ++i) {
                double v = InnerProduct(vec[i], w.data(), dim);
                if (v > 0)
                    label_prediction[i][a] ++;
                else
                    label_prediction[i][b] ++;
            }
        }

    double ave_f1 = 0;
    for (int a = 0; a < test.card; ++a) {
        std::vector<double> p, n;
        for (int i = 0; i < test.size; ++i)
            if (test.label[i] == a)
                p.push_back(label_prediction[i][a]);
            else if (test.label[i] != -1)
                n.push_back(label_prediction[i][a]);
        ave_f1 += EvaluateF1(p, n);
    }
    ave_f1 /= train.card;
    return ave_f1;
}

double EvaluateF1LabelPropagation(Model* model, const Label& test) {
    double ave_f1 = 0;
    for (int a = 0; a < test.card; ++a) {
        std::vector<double> p, n;
        for (int i = 0; i < test.size; ++i)
            if (test.label[i] == a)
                p.push_back(model->GetEmbedding(i)[a]);
            else if (test.label[i] != -1)
                n.push_back(model->GetEmbedding(i)[a]);
        ave_f1 += EvaluateF1(p, n);
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
