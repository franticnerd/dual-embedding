#include "base.h"
#include <set>

class CommonNeighbor : public Model {
    Graph base;
  public:
    CommonNeighbor(const Graph& base_) : base(base_) {}
    double Evaluate(int x, int y) {
        std::set<int> set;
        for (const auto& p : base.edge[x])
            set.insert(p);
        double val = 0;
        for (const auto& p : base.edge[y])
            if (set.count(p) > 0)
                val++;
        return val;
    }
};

Model* GetCommonNeighbor(const Graph& base) {
    return new CommonNeighbor(base);
}