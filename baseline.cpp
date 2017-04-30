#include "base.h"
#include "utility.h"

#include <set>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>

namespace {
    std::mt19937 gen;
}   // anonymous namespace

class CommonNeighbor : public Model {
    Graph base_;
    const double normalizer_;
  public:
    CommonNeighbor(const Graph& base, double normalizer) : base_(base), normalizer_(normalizer) {}
    double Evaluate(int x, int y) {
        std::set<int> set;
        for (const auto& p : base_.edge[x])
            set.insert(p);
        double val = 0;
        for (const auto& p : base_.edge[y])
            if (set.count(p) > 0)
                val++;
        if (set.count(y) > 0) 
            val += sqrt(base_.edge[x].size()) + sqrt(base_.edge[y].size());
        return val / normalizer_;
    }
};

class AdamicAdar : public Model {
    Graph base_;
    std::vector<int> cnt_;
  public:
    AdamicAdar(const Graph& base) : base_(base), cnt_(base.size, 0) {}
    double Evaluate(int x, int y) {
        for (int p : base_.edge[x])
            cnt_[p] = 1;
        double val = 0;
        for (int p : base_.edge[y])
            if (cnt_[p] == 1)
                val += 1 / log(base_.edge[p].size());
        for (int p : base_.edge[x])
            cnt_[p] = 0;
        return val;
    }
};

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
    const std::vector<double>& GetEmbedding(int x) { return embedding[x]; }
};

Model* GetCommonNeighbor(const Graph& base, double normalizer) {
    return new CommonNeighbor(base, normalizer);
}

Model* GetAdamicAdar(const Graph& base) {
    return new AdamicAdar(base);
}

Model* GetPredefined(const std::string& filename) {
    return new Predefined(filename);
}

Model* GetRandom() {
    return new Random();
}