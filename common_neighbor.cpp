#include "base.h"
#include <set>
#include <cmath>

class CommonNeighbor : public Model {
    Graph base_;
  public:
    CommonNeighbor(const Graph& base) : base_(base) {}
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
        return val / 120;
    }
};

Model* GetCommonNeighbor(const Graph& base) {
    return new CommonNeighbor(base);
}