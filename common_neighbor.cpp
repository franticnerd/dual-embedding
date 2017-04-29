#include "base.h"
#include <set>
#include <vector>
#include <cmath>

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

Model* GetCommonNeighbor(const Graph& base, double normalizer) {
    return new CommonNeighbor(base, normalizer);
}

Model* GetAdamicAdar(const Graph& base) {
    return new AdamicAdar(base);
}