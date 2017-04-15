#include "unit_test.h"
#include "base.h"

#include <cassert>
#include <memory>

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
    std::unique_ptr<Model> model(GetFiniteEmbedding(graph, 5));
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
}   

void KernelEmbeddingTest() {
    Graph graph(7);
    MakeGraph(&graph);
    std::unique_ptr<Model> model(GetKernelEmbedding(graph));
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
}

void SparseEmbeddingTest() {
    Graph graph(7);
    MakeGraph(&graph);
    std::unique_ptr<Model> model(GetSparseEmbedding(graph));
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
}

void EmbeddingTest() {
    FiniteEmbeddingTest();
    KernelEmbeddingTest();
    SparseEmbeddingTest();
}