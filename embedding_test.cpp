#include "unit_test.h"
#include "base.h"

#include <cassert>
#include <memory>
#include <iostream>

void MakeGraph(Graph* graph) {
    *graph = Graph(7);
    graph->AddEdge(0, 1);
    graph->AddEdge(0, 2);
    graph->AddEdge(1, 3);
    graph->AddEdge(2, 3);
    graph->AddEdge(3, 4);
    graph->AddEdge(4, 5);
    graph->AddEdge(4, 6);
    graph->AddEdge(5, 6);
}

void FiniteEmbeddingTest() {
    Graph graph(7);
    MakeGraph(&graph);
    Graph negative(7);
    SampleNegativeGraphPreferential(graph, &negative, 1);
    std::unique_ptr<Model> model(GetFiniteEmbedding(graph, negative, 5));
    std::cout << model->Evaluate(1, 2) << " " << model->Evaluate(2, 6) << " " << model->Evaluate(1, 5) << "\n";
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
}   

void KernelEmbeddingTest() {
    Graph graph(7);
    MakeGraph(&graph);
    Graph negative(7);
    SampleNegativeGraphPreferential(graph, &negative, 1);
    std::unique_ptr<Model> model(GetKernelEmbedding(graph, negative));
    std::cout << model->Evaluate(1, 2) << " " << model->Evaluate(2, 6) << " " << model->Evaluate(1, 5) << "\n";
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
}

void SparseEmbeddingTest() {
    Graph graph(7);
    MakeGraph(&graph);
    Graph negative(7);
    //SampleNegativeGraph(graph, &negative);
    std::unique_ptr<Model> model(GetSparseEmbedding(graph, negative));
    std::cout << model->Evaluate(1, 2) << " " << model->Evaluate(2, 6) << " " << model->Evaluate(1, 5) << " " << model->Evaluate(1, 4) << "\n";
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 4));
}

void CommonNeighborTest() {
    Graph graph(7);
    MakeGraph(&graph);
    Graph negative(7);
    SampleNegativeGraphPreferential(graph, &negative, 1);
    std::unique_ptr<Model> model(GetCommonNeighbor(graph));
    std::cout << model->Evaluate(1, 2) << " " << model->Evaluate(2, 6) << " " << model->Evaluate(1, 5) << "\n";
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
}

void EmbeddingTest() {
    FiniteEmbeddingTest();
    KernelEmbeddingTest();
    SparseEmbeddingTest();
    CommonNeighborTest();
}