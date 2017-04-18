#include "utility.h"
#include "unit_test.h"
#include <cassert>

void F1Test() {
    std::vector<double> pos, neg;
    pos.push_back(2); pos.push_back(3);
    neg.push_back(0); neg.push_back(1); neg.push_back(2);
    assert(fabs(EvaluateF1(pos, neg) - 0.8) < 0.01);
}

void MAPTest() {
    std::vector<double> pos, neg;
    pos.push_back(2); pos.push_back(3);
    neg.push_back(0); neg.push_back(1); neg.push_back(2);
    assert(fabs(EvaluateMAP(pos, neg) - 0.83) < 0.01);
}

void UtilityTest() {
    F1Test();
    MAPTest();
}