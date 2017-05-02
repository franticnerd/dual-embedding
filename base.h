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
    Graph() : size(0) {}
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

struct Label {
    int card, size;
    std::vector<int> label;
    Label() {}
    Label(int size_) : card(0), size(size_), label(size_, -1) {}
    void SetLabel(int x, int l) { label[x] = l; card = (l >= card ? l + 1 : card); }
};

class Model {
    std::vector<double> null_;
  public:
    Model() {}
    virtual ~Model() {}
    virtual double Evaluate(int x, int y) { return 0; }
    virtual const std::vector<double>& GetEmbedding(int x) { return null_; }
};

Model* GetFiniteEmbedding(const Graph& postive, const Graph& negative, int dimension, double neg_penalty, double regularizer, double deg_norm_pow);
Model* GetSequentialFiniteEmbedding(const Graph& positive, const Graph& negative, int dimension, double neg_penalty, double regularizer);
Model* GetFiniteContrastEmbedding(const Graph& positive, const Graph& negative, int sample_ratio, int dimension, double regularizer, double deg_norm_pow);
Model* GetKernelEmbedding(const Graph& postive, const Graph& negative, double neg_penalty, double regularizer, double deg_norm_pow);
Model* GetSparseEmbedding(const Graph& postive, const Graph& negative, double neg_penalty, double regularizer, double deg_norm_pow);
Model* GetCommonNeighbor(const Graph& base, double normalizer);
Model* GetAdamicAdar(const Graph& base);
Model* GetPredefined(const std::string& filename);
Model* GetRandom();
Model* GetLabelPropagation(const Graph& base, const Label& label);

void SampleNegativeGraphUniform(const Graph& positive, Graph* negative);
void SampleNegativeGraphPreferential(const Graph& positive, Graph* negative, double p);
void SampleNegativeGraphLocal(const Graph& positive, Graph* negative);
void RemoveRedundant(const Graph& positive, Graph* negative);

double EvaluateAveragePrecision(Model* model, const Graph& pos, const Graph& neg);
double EvaluateF1(Model* model, const Label& train, const Label& test, double regularizer, int sample_ratio);
double EvaluateF1LabelPropagation(Model* model, const Label& test);
void EvaluateAll(Model* model, const Graph& train_pos, const Graph& train_neg, const Graph& test_pos, const Graph& test_neg);

void ReadDataset(const std::string& nodefile, const std::string& edgefile, Graph* graph);
void ReadLabel(const std::string& nodefile, const std::string& labelfile, Label* label);

