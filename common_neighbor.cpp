#include "base.h"
#include <set>

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
        return val;
    }
};

Model* GetCommonNeighbor(const Graph& base) {
    return new CommonNeighbor(base);
}