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

double EvaluateF1(const std::vector<double>& positive, const std::vector<double>& negative) {
    if (positive.size() == 0) return 0;

    std::vector<double> combine;
    for (double p : positive)
        combine.push_back(p);
    for (double p : negative)
        combine.push_back(p);
    sort(combine.begin(), combine.end());
    double cutoff = combine[0], eps = 1e-4;
    for (int i = combine.size() - 1; i >= (int)negative.size(); --i)
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
    if (precision < 1e-5 || recall < 1e-5)
        return 0;
    return (precision * recall) / (precision + recall) * 2;
}

double EvaluateAveragePrecision(const std::vector<double>& positive, const std::vector<double>& negative) {
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

Prop_Sampler::Prop_Sampler(const std::vector<double>& x) {
    double sum = 0;
    for (double item : x) {
        sum += item;
        ps.push_back(sum);
    }
}

int Prop_Sampler::GetSample() {
    std::uniform_real_distribution<double> dist(0, 1);
    double val = ps.back() * dist(gen);
    return std::lower_bound(ps.begin(), ps.end(), val) - ps.begin();
}

int prop_sample(const std::vector<double>& x) {
    std::uniform_real_distribution<double> dist(0, 1);
    double sum = 0;
    for (int i = 0; i < (int)x.size(); ++i)
        sum += x[i];
    sum *= dist(gen);
    for (int i = 0; i < (int)x.size(); ++i)
        if (sum < x[i])
            return i;
        else
            sum -= x[i];
    return 0;
}
