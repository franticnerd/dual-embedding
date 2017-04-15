#pragma once

#include <vector>

struct Matrix {
    int m, n;
    std::vector<double> val;
    Matrix(int m_, int n_) : m(m_), n(n_) {}
    double& At(int i, int j) { return val[i * m + j]; }
};

struct Graph {
    int size;
    std::vector<std::vector<int>> edge;
    Graph(int size_) : size(size_) {
        edge.resize(size);
    }
    void AddEdge(int x, int y) {
        edge[x].push_back(y);
        edge[y].push_back(x);
    }
};

struct Edge {
    int x, y;
    Edge(int x_, int y_) : x(x_), y(y_) {}
};

class Model {
  public:
    Model() {}
    virtual ~Model() {}
    virtual double Evaluate(int x, int y) = 0;
};

void ReadDataset(const std::string& nodefile, const std::string& edgefile, Graph* graph);
Model* GetFiniteEmbedding(const Graph& graph, int dimension);
Model* GetKernelEmbedding(const Graph& graph);
Model* GetSparseEmbedding(const Graph& graph);
void SampleNegativeGraph(const Graph& positive, Graph* negative);
void SampleLocalNegativeGraph(const Graph& positive, Graph* negative);

Model* GetCommonNeighbor(const Graph& base);