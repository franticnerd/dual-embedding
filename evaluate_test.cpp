#include "base.h"
#include "unit_test.h"
#include <cassert>
#include <memory>
#include <iostream>

void MakeGraphLabel(Graph* graph, Label* train, Label* test) {
    *graph = Graph(7);
    graph->AddEdge(0, 1);
    graph->AddEdge(0, 2);
    graph->AddEdge(1, 3);
    graph->AddEdge(2, 3);
    graph->AddEdge(3, 4);
    graph->AddEdge(4, 5);
    graph->AddEdge(4, 6);
    graph->AddEdge(5, 6);
    *train = Label(7);
    train->SetLabel(0, 0);
    train->SetLabel(3, 0);
    train->SetLabel(4, 1);
    *test = Label(7);
    test->SetLabel(1, 0);
    test->SetLabel(2, 0);
    test->SetLabel(5, 1);
    test->SetLabel(6, 1);
}

void EvaluateF1Test() {
    Graph graph;
    Label train, test;
    MakeGraphLabel(&graph, &train, &test);
    Graph negative(7);
    SampleNegativeGraphUniform(graph, &negative);
    RemoveRedundant(graph, &negative);
    std::unique_ptr<Model> model(GetFiniteEmbedding(graph, negative, 5, 0.2, 1, 0));
    std::cout << EvaluateF1(model.get(), train, test, 1, 2) << "\n";
    assert(fabs(EvaluateF1(model.get(), train, test, 1, 2) - 1) < 0.01);
}

void EvaluateF1LabelPropagationTest() {
    Graph graph;
    Label train, test;
    MakeGraphLabel(&graph, &train, &test);
    std::unique_ptr<Model> model(GetLabelPropagation(graph, train));
    std::cout << EvaluateF1LabelPropagation(model.get(), test) << "\n";
    assert(fabs(EvaluateF1LabelPropagation(model.get(), test) - 1) < 0.01);
}

void EvaluateTest() {
    EvaluateF1Test();
    EvaluateF1LabelPropagationTest();
}