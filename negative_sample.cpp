#include "base.h"
#include "utility.h"
#include <random>
#include <cmath>
#include <set>

#define NEGATIVE_RATIO 2

namespace {
    std::mt19937 gen;
}   // anonymous namespace

void SampleNegativeGraphUniform(const Graph& positive, Graph* negative) {
    std::uniform_int_distribution<int> dist(0, positive.size - 1);
    for (int i = 0; i < positive.size; ++i)
        for (int j = 0; j < (int)positive.edge[i].size() * NEGATIVE_RATIO; ++j) {
            int targetA = dist(gen);
            int targetB = dist(gen);
            if (targetA != targetB)
                negative->AddEdge(targetA, targetB);
        }
}

void SampleNegativeDGraphUniform(const DGraph& positive, DGraph* negative) {
    std::uniform_int_distribution<int> dist(0, positive.size - 1);
    for (int i = 0; i < positive.size; ++i)
        for (int j = 0; j < (int)positive.out_edge[i].size() * NEGATIVE_RATIO * 2; ++j) {
            int targetA = dist(gen);
            int targetB = dist(gen);
            if (targetA != targetB)
                negative->AddEdge(targetA, targetB);
        }

}

void SampleNegativeGraphPreferential(const Graph& positive, Graph* negative, double p) {
    std::vector<double> rate;
    for (int i = 0; i < positive.size; ++i)
        rate.push_back(pow(positive.edge[i].size(), p));
    Prop_Sampler sampler(rate);
    for (int i = 0; i < positive.size; ++i)
        for (int j = 0; j < (int)positive.edge[i].size() * NEGATIVE_RATIO; ++j) {
            int targetA = sampler.GetSample();
            int targetB = sampler.GetSample();
            if (targetA != targetB) 
                negative->AddEdge(targetA, targetB);
        }
}

void SampleNegativeGraphLocal(const Graph& positive, Graph* negative) {
    std::uniform_real_distribution<double> dist(0, 1);
    for (int i = 0; i < positive.size; ++i)
        for (int t : positive.edge[i])
            for (int j = 0; j < NEGATIVE_RATIO; ++j) {
                int target = positive.edge[t][(int)(dist(gen) * positive.edge[t].size())];
                if (target != i)
                    negative->AddEdge(i, target);
            }
}

void RemoveRedundant(const Graph& positive, Graph* negative) {
    std::set<std::pair<int, int>> pr;
    for (int i = 0; i < positive.size; ++i)
        for (int j : positive.edge[i])
            pr.insert(std::make_pair(i, j));
    for (int i = 0; i < negative->size; ++i) {
        std::vector<int> new_edge;
        for (int j : negative->edge[i])
            if (pr.count(std::make_pair(i, j)) == 0)
                new_edge.push_back(j);
        negative->edge[i] = new_edge;
    }
}

void RemoveRedundant(const DGraph& positive, DGraph* negative) {
    std::set<std::pair<int, int>> pr;
    for (int i = 0; i < positive.size; ++i)
        for (int j : positive.out_edge[i])
            pr.insert(std::make_pair(i, j));
    for (int i = 0; i < negative->size; ++i) {
        std::vector<int> new_edge;
        for (int j : negative->out_edge[i])
            if (pr.count(std::make_pair(i, j)) == 0)
                new_edge.push_back(j);
        negative->out_edge[i] = std::move(new_edge);
        new_edge.clear();
        for (int j : negative->in_edge[i])
            if (pr.count(std::make_pair(j, i)) == 0)
                new_edge.push_back(j);
        negative->in_edge[i] = std::move(new_edge);
    }

}