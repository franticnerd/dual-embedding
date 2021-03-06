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

void MakeDGraph(DGraph* graph) {
    *graph = DGraph(8);
    graph->AddEdge(0, 4);
    graph->AddEdge(0, 6);
    graph->AddEdge(0, 7);
    graph->AddEdge(1, 5);
    graph->AddEdge(1, 6);
    graph->AddEdge(1, 7);
    graph->AddEdge(2, 4);
    graph->AddEdge(2, 5);
    graph->AddEdge(2, 7);
    graph->AddEdge(3, 5);
    graph->AddEdge(3, 4);
    graph->AddEdge(3, 6);
}

void FiniteEmbeddingTest() {
    Graph graph(7);
    MakeGraph(&graph);
    Graph negative(7);
    SampleNegativeGraphUniform(graph, &negative);
    RemoveRedundant(graph, &negative);
    std::unique_ptr<Model> model(GetFiniteEmbedding(graph, negative, 5, 0.2, 1));
    std::cout << model->Evaluate(1, 2) << " " << model->Evaluate(2, 6) << " " << model->Evaluate(1, 5) << "\n";
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
}   

void FiniteContrastEmbeddingTest() {
    Graph graph(7);
    MakeGraph(&graph);
    Graph negative(7);
    SampleNegativeGraphUniform(graph, &negative);
    RemoveRedundant(graph, &negative);
    std::unique_ptr<Model> model(GetFiniteContrastEmbedding(graph, negative, 3, 5, 1));
    std::cout << model->Evaluate(1, 2) << " " << model->Evaluate(2, 6) << " " << model->Evaluate(1, 5) << "\n";
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
}

void DirectedFiniteEmbeddingTest() {
    DGraph graph(8);
    MakeDGraph(&graph);
    DGraph negative(8);
    SampleNegativeDGraphUniform(graph, &negative);
    RemoveRedundant(graph, &negative);
    std::unique_ptr<Model> model(GetDirectedFiniteEmbedding(graph, negative, 5, 0.2, 1));
    std::cout << model->Evaluate(1, 4) << " " << model->Evaluate(1, 3) << " " << model->Evaluate(5, 1) << "\n";
    assert(model->Evaluate(1, 4) > model->Evaluate(1, 3));
    assert(model->Evaluate(1, 4) > model->Evaluate(5, 1));
}

void DirectedFiniteContrastEmbeddingTest() {
    DGraph graph(8);
    MakeDGraph(&graph);
    DGraph negative(8);
    SampleNegativeDGraphUniform(graph, &negative);
    RemoveRedundant(graph, &negative);
    std::unique_ptr<Model> model(GetDirectedFiniteContrastEmbedding(graph, negative, 3, 5, 1));
    std::cout << model->Evaluate(0, 5) << " " << model->Evaluate(0, 3) << " " << model->Evaluate(5, 1) << "\n";
    assert(model->Evaluate(0, 5) > model->Evaluate(0, 3));
    assert(model->Evaluate(0, 5) > model->Evaluate(5, 1));
}

void KernelEmbeddingTest() {
    Graph graph(7);
    MakeGraph(&graph);
    Graph negative(7);
    SampleNegativeGraphUniform(graph, &negative);
    RemoveRedundant(graph, &negative);
    std::unique_ptr<Model> model(GetKernelEmbedding(graph, negative, 0.2, 1));
    std::cout << model->Evaluate(1, 2) << " " << model->Evaluate(2, 6) << " " << model->Evaluate(1, 5) << "\n";
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
}

void SparseEmbeddingTest() {
    Graph graph(7);
    MakeGraph(&graph);
    Graph negative(7);
    //SampleNegativeGraph(graph, &negative);
    std::unique_ptr<Model> model(GetSparseEmbedding(graph, negative, 1, 1));
    std::cout << model->Evaluate(1, 2) << " " << model->Evaluate(2, 6) << " " << model->Evaluate(1, 5) << " " << model->Evaluate(1, 4) << "\n";
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 4));
}

void CommonNeighborTest() {
    Graph graph(7);
    MakeGraph(&graph);
    Graph negative(7);
    SampleNegativeGraphUniform(graph, &negative);
    RemoveRedundant(graph, &negative);
    std::unique_ptr<Model> model(GetCommonNeighbor(graph, 5));
    std::cout << model->Evaluate(1, 2) << " " << model->Evaluate(2, 6) << " " << model->Evaluate(1, 5) << "\n";
    assert(model->Evaluate(1, 2) > model->Evaluate(2, 6));
    assert(model->Evaluate(1, 2) > model->Evaluate(1, 5));
}

void EmbeddingTest() {
    FiniteEmbeddingTest();
    FiniteContrastEmbeddingTest();
    KernelEmbeddingTest();
    SparseEmbeddingTest();
    DirectedFiniteEmbeddingTest();
    DirectedFiniteContrastEmbeddingTest();
    CommonNeighborTest();
}