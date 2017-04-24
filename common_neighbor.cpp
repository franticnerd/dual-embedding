#include "base.h"
#include <set>
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

Model* GetCommonNeighbor(const Graph& base, double normalizer) {
    return new CommonNeighbor(base, normalizer);
}