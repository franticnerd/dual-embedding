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

void ReadDataset(const std::string& nodefile, const std::string& edgefile, Graph* train, Graph* test);
Model* GetFiniteEmbedding(const Graph& postive, const Graph& negative, int dimension, double neg_penalty, double regularizer, double deg_norm_pow);
Model* GetSequentialFiniteEmbedding(const Graph& positive, const Graph& negative, int dimension, double neg_penalty, double regularizer);
Model* GetFiniteContrastEmbedding(const Graph& positive, const Graph& negative, int sample_ratio, int dimension, double regularizer, double deg_norm_pow);
Model* GetKernelEmbedding(const Graph& postive, const Graph& negative, double neg_penalty, double regularizer, double deg_norm_pow);
Model* GetSparseEmbedding(const Graph& postive, const Graph& negative, double neg_penalty, double regularizer, double deg_norm_pow);
void SampleNegativeGraphUniform(const Graph& positive, Graph* negative);
void SampleNegativeGraphPreferential(const Graph& positive, Graph* negative, double p);
void SampleNegativeGraphLocal(const Graph& positive, Graph* negative);
void RemoveRedundant(const Graph& positive, Graph* negative);

Model* GetCommonNeighbor(const Graph& base, double normalizer);