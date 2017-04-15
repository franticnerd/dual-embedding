#include "utility.h"
#include <vector>
#include <random>
#include <algorithm>

namespace {
    std::mt19937 gen(910109);
}   // anonymous namespace

void RandomPermutation(std::vector<int>* vec) {
    std::uniform_int_distribution<int> dist(0, vec->size() - 1);
    for (int i = 0; i < (int)vec->size(); ++i) {
        int t = dist(gen);
        std::swap(vec->at(i), vec->at(t));
    }
}

double InnerProduct(const std::vector<double>& x, const std::vector<double>& y) {
    double val = 0;
    for (int i = 0; i < (int)x.size(); ++i)
        val += x[i] * y[i];
    return val;
}

double EvaluateF1(const std::vector<double>& positive, const std::vector<double>& negative) {
    if (positive.size() == 0) return 0;

    std::vector<double> combine;
    for (double p : positive)
        combine.push_back(p);
    for (double p : negative)
        combine.push_back(p);
    sort(combine.begin(), combine.end());
    double cutoff, eps = 1e-4;
    for (int i = combine.size() - 1; i >= negative.size(); --i)
        if (combine[i] >= combine[0] + eps)
            cutoff = combine[i];

    int pos_in_range = 0, neg_in_range = 0;
    for (double p : positive)
        if (p >= cutoff - eps)
            ++pos_in_range;
    for (double p : negative)
        if (p >= cutoff - eps)
            ++neg_in_range;
    double precision = (double)pos_in_range / (double)(pos_in_range + neg_in_range);
    double recall = (double)pos_in_range / (double)positive.size();
    return (precision * recall) / (precision + recall) * 2;
}

double EvaluateMAP(const std::vector<double>& positive, const std::vector<double>& negative) {
    std::vector<std::pair<double, int>> combine;
    for (double p : positive)
        combine.push_back(std::make_pair(-p, 1));
    for (double p : negative)
        combine.push_back(std::make_pair(-p, -1));
    sort(combine.begin(), combine.end());
    int span_pos = 0, span_neg = 0, tot_pos = 0, tot_neg = 0;
    double current_p = -1e6, eps = 1e-4, ave_p = 0;
    for (const auto& pair : combine) {
        if (pair.first > current_p + eps) {
            tot_pos += span_pos;
            tot_neg += span_neg;
            if (span_pos > 0)
                ave_p += (double)span_pos / (double)positive.size() * tot_pos / (double)(tot_pos + tot_neg);
            span_pos = 0;
            span_neg = 0;
            current_p = pair.first;
        }
        if (pair.second == 1)
            ++span_pos;
        else
            ++span_neg;
    }
    ave_p += (double)span_pos / (double)positive.size() * positive.size() / (double)(positive.size() + negative.size());
    return ave_p;
}