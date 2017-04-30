#include "base.h"

#include <memory>
#include <iostream>

struct EvaluateConfig {
    Graph train, neg_train, test, neg_test;
    Label train_label, test_label;
    bool predict_edge, predict_label;

    // Finite Embedding parameters
    int finite_dim;
    double finite_neg_penalty, finite_regularizer;

    // Finite Contrast parameters
    int finite_contrast_dim, finite_contrast_sample_ratio;
    double finite_contrast_regularizer;

    // Kernel parameters
    double kernel_neg_penalty, kernel_regularizer;

    // Sparse parameters
    double sparse_neg_penalty, sparse_regularizer;
    
    // Sequential parameters
    int sequential_dim;
    double sequential_neg_penalty, sequential_regularizer;

    // Common Neighbor parameters
    double normalizer;

    // Predefined parameters
    std::string filename;
};

void EvalFiniteEmbedding(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Finite Embedding\n";
    model.reset(GetFiniteEmbedding(config.train, config.neg_train, config.finite_dim, config.finite_neg_penalty, config.finite_regularizer, 0));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.train, config.neg_train, config.test, config.neg_test);
    }
    if (config.predict_label) {
        std::cout << "Evaluating Label Prediction\n";
        std::cout << "Average F1:" << EvaluateF1(model.get(), config.train_label, config.test_label, 1) << "\n";
    }
}

void EvalFiniteContrastEmbedding(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Finite Contrast Embedding\n";
    model.reset(GetFiniteContrastEmbedding(config.train, config.neg_train, config.finite_contrast_sample_ratio, config.finite_contrast_dim, config.finite_contrast_regularizer, 0));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.train, config.neg_train, config.test, config.neg_test);
    }
    if (config.predict_label) {
        std::cout << "Evaluating Label Prediction\n";
        std::cout << "Average F1:" << EvaluateF1(model.get(), config.train_label, config.test_label, 1) << "\n";
    }
}

void EvalKernelEmbedding(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Kernel Embedding\n";
    model.reset(GetKernelEmbedding(config.train, config.neg_train, config.kernel_neg_penalty, config.kernel_regularizer, 0));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.train, config.neg_train, config.test, config.neg_test);
    }
}

void EvalSparseEmbedding(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    Graph neg_empty(config.train.size);
    std::cout << "Training Sparse Embedding\n";
    model.reset(GetSparseEmbedding(config.train, neg_empty, config.sparse_neg_penalty, config.sparse_regularizer, 0));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.train, config.neg_train, config.test, config.neg_test);
    }
}

void EvalSequentialEmbedding(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Sequential Embedding\n";
    model.reset(GetSequentialFiniteEmbedding(config.train, config.neg_train, config.sequential_dim, config.sequential_neg_penalty, config.sequential_regularizer));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.train, config.neg_train, config.test, config.neg_test);
    }
}

void EvalCommonNeighbor(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Common Neighbor\n";
    model.reset(GetCommonNeighbor(config.train, config.normalizer));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.train, config.neg_train, config.test, config.neg_test);
    }
}

void EvalPredefined(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Predefined\n";
    model.reset(GetPredefined(config.filename));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        std::cout << EvaluateAveragePrecision(model.get(), config.test, config.neg_test) << "\n";
    }
}

void EvalLabelPropagation(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Label Propagation\n";
    model.reset(GetLabelPropagation(config.train, config.train_label));
    if (config.predict_label) {
        std::cout << "Evaluating Label Prediction\n";
        std::cout << EvaluateF1LabelPropagation(model.get(), config.test_label) << "\n";
    }
}

void EvalRandom(const EvaluateConfig& config) {
    std::cout << "Training Random\n";
    std::unique_ptr<Model> model(GetRandom());
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        std::cout << EvaluateAveragePrecision(model.get(), config.test, config.neg_test) << "\n";
    }
}

int main() {
    int test_case = 1;

    EvaluateConfig config;
    std::cout << "Reading Dataset\n";
    switch (test_case) {
    case 0:
        ReadDataset("node-w-filter.txt", "edge-ww-train-filter.txt", &config.train);
        ReadDataset("node-w-filter.txt", "edge-ww-val-filter.txt", &config.test);
        config.predict_edge = true;
        config.predict_label = false;

        config.finite_dim = 100; config.finite_neg_penalty = 0.03; config.finite_regularizer = 30;
        config.finite_contrast_sample_ratio = 6; config.finite_contrast_dim = 100; config.finite_contrast_regularizer = 100;
        config.kernel_neg_penalty = 0.03; config.kernel_regularizer = 30;
        config.sparse_neg_penalty = 0.015; config.sparse_regularizer = 15;
        config.sequential_dim = 100; config.sequential_neg_penalty = 0.1; config.sequential_regularizer = 2;
        config.normalizer = 120;
        config.filename = "vec-filter.txt";
        break;
    case 1:
        ReadDataset("blog-node.txt", "blog-edge.txt", &config.train);
        ReadLabel("blog-node.txt", "blog-label-train.txt", &config.train_label);
        ReadLabel("blog-node.txt", "blog-label-test.txt", &config.test_label);
        config.predict_edge = false;
        config.predict_label = true;

        config.finite_dim = 100; config.finite_neg_penalty = 0.03; config.finite_regularizer = 30;
        config.finite_contrast_sample_ratio = 6; config.finite_contrast_dim = 100; config.finite_contrast_regularizer = 100;
        config.kernel_neg_penalty = 0.03; config.kernel_regularizer = 30;
        config.sparse_neg_penalty = 0.015; config.sparse_regularizer = 15;
        config.sequential_dim = 100; config.sequential_neg_penalty = 0.1; config.sequential_regularizer = 2;
        break;
    }

    std::cout << "Sampling Negative Dataset\n";
    config.neg_train = Graph(config.train.size);
    SampleNegativeGraphUniform(config.train, &config.neg_train);
    
    config.neg_test = Graph(config.test.size);
    //SampleNegativeGraphPreferential(test, &neg_test, 1);    
    SampleNegativeGraphUniform(config.test, &config.neg_test);
    
    RemoveRedundant(config.train, &config.neg_test);
    RemoveRedundant(config.test, &config.neg_test);

    EvalFiniteEmbedding(config);
    EvalFiniteContrastEmbedding(config);
    //EvalKernelEmbedding(train, neg_train, test, neg_test);
    //EvalSparseEmbedding(train, neg_empty, test, neg_test);
    //EvalSequentialEmbedding(train, neg_train, test, neg_test);
    //EvalCommonNeighbor(config);
    //EvalPredefined(config);
    EvalLabelPropagation(config);
    //EvalRandom(config);

    system("pause");
}