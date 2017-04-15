#include "base.h"
#include "utility.h"
#include <random>

#define NEGATIVE_RATIO 2

namespace {
    std::mt19937 gen;
}   // anonymous namespace

void SampleNegativeGraph(const Graph& positive, Graph* negative) {
    std::vector<double> rate;
    for (int i = 0; i < positive.size; ++i)
        rate.push_back(positive.edge[i].size());
    Prop_Sampler sampler(rate);
    for (int i = 0; i < positive.size; ++i)
        for (int j = 0; j < (int)positive.edge[i].size() * NEGATIVE_RATIO; ++j) {
            int target = sampler.GetSample();
            if (target != i) 
                negative->AddEdge(i, target);
        }
}

void SampleLocalNegativeGraph(const Graph& positive, Graph* negative) {
    std::uniform_real_distribution<double> dist(0, 1);
    for (int i = 0; i < positive.size; ++i)
        for (int t : positive.edge[i])
            for (int j = 0; j < NEGATIVE_RATIO; ++j) {
                int target = positive.edge[t][(int)(dist(gen) * positive.edge[t].size())];
                if (target != i)
                    negative->AddEdge(i, target);
            }
}