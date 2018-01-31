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

struct DGraph {
    int size;
    std::vector<std::vector<int>> out_edge, in_edge;
    DGraph() : size(0) {}
    DGraph(int size_) : size(size_) {
        in_edge.resize(size);
        out_edge.resize(size);
    }
    void AddEdge(int x, int y) {
        out_edge[x].push_back(y);
        in_edge[y].push_back(x);
    }
};

struct Edge {
    int x, y;
    Edge(int x_, int y_) : x(x_), y(y_) {}
};

struct SingleLabel {
    int size;
    std::vector<int> label;
    SingleLabel(int size_) : size(size_), label(size_, -1) {}
    void SetLabel(int x, int l) {
        label[x] = l;
    }
};

struct Label {
    int size, card;
    std::vector<std::vector<int>> label_instance;
    std::vector<bool> labeled;
    Label() {}
    Label(int size_) : size(size_), labeled(size_, false) {}
    void SetLabel(int x, int l) {
        if (l >= (int)label_instance.size()) {
            label_instance.resize(l + 1);
            card = l + 1;
        }
        label_instance[l].push_back(x);
        labeled[x] = true;
    }
};

class Model {
    std::vector<double> null_;
  public:
    Model() {}
    virtual ~Model() {}
    virtual double Evaluate(int x, int y) { return 0; }
    virtual const std::vector<double>& GetEmbedding(int x) { return null_; }
};

Model* GetFiniteEmbedding(const Graph& postive, const Graph& negative, int dimension, double neg_penalty, double regularizer);
Model* GetFiniteSGD(const Graph& postive, const Graph& negative, int dimension, double neg_penalty, double regularizer);
Model* GetSequentialFiniteEmbedding(const Graph& positive, const Graph& negative, int dimension, double neg_penalty, double regularizer);
Model* GetFiniteContrastEmbedding(const Graph& positive, const Graph& negative, int sample_ratio, int dimension, double regularizer);
Model* GetKernelEmbedding(const Graph& postive, const Graph& negative, double neg_penalty, double regularizer);
Model* GetSparseEmbedding(const Graph& postive, const Graph& negative, double neg_penalty, double regularizer);
Model* GetDirectedFiniteEmbedding(const DGraph& graph, const DGraph& negative, int dimension, double neg_penalty, double regularizer);
Model* GetDirectedFiniteContrastEmbedding(const DGraph& graph, const DGraph& negative, int sample_ratio, int dimension, double regularizer);
Model* GetCommonNeighbor(const Graph& base, double normalizer);
Model* GetAdamicAdar(const Graph& base);
Model* GetPredefined(const std::string& node_file, const std::string& embedding_file);
Model* GetRandom();
Model* GetLabelPropagation(const Graph& base, const SingleLabel& label);
Model* GetSVD(const std::string& node_file, const std::string& u_file, const std::string& sv_file, const std::string& v_file);

void SampleNegativeGraphUniform(const Graph& positive, Graph* negative);
void SampleNegativeDGraphUniform(const DGraph& positive, DGraph* negative);
void SampleNegativeGraphPreferential(const Graph& positive, Graph* negative, double p);
void SampleNegativeGraphLocal(const Graph& positive, Graph* negative);
void RemoveRedundant(const Graph& positive, Graph* negative);
void RemoveRedundant(const DGraph& positive, DGraph* negative);

double EvaluatePredictedAP(Model* model, const Graph& train, const Graph& pos, const Graph& neg, double regularizer, int sample_ratio);
double EvaluateAveragePrecision(Model* model, const Graph& pos, const Graph& neg);
double EvaluateAveragePrecision(Model* model, const DGraph& pos, const DGraph& neg);
double EvaluateF1(Model* model, const Label& train, const Label& test, double regularizer, int sample_ratio, bool normalize);
double EvaluateF1LabelPropagation(const Graph& base, const Label& train, const Label& test);
void EvaluateAll(Model* model, const Graph& train_pos, const Graph& train_neg, const Graph& test_pos, const Graph& test_neg);
void EvaluateAll(Model* model, const DGraph& train_pos, const DGraph& train_neg, const DGraph& test_pos, const DGraph& test_neg);

void ReadDataset(const std::string& nodefile, const std::string& edgefile, Graph* graph);
void ReadDirectedDataset(const std::string& nodefile, const std::string& edgefile, DGraph* graph);
void ReadLabel(const std::string& nodefile, const std::string& labelfile, Label* label);

