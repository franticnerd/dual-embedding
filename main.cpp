#include "base.h"

#include <memory>
#include <iostream>

struct EvaluateConfig {
    Graph train, neg_train, test, neg_test;
    DGraph d_train, d_neg_train, d_test, d_neg_test;
    Label train_label, test_label;
    bool predict_edge, predict_label;

    // Finite Embedding parameters
    int finite_dim;
    double finite_neg_penalty, finite_regularizer;

    // Finite Contrast parameters
    int finite_contrast_dim, finite_contrast_sample_ratio;
    double finite_contrast_regularizer;

    // Directed Finite Embedding parameters
    int d_finite_dim;
    double d_finite_neg_penalty, d_finite_regularizer;

    // Directed Finite Contrast parameters
    int d_finite_contrast_dim, d_finite_contrast_sample_ratio;
    double d_finite_contrast_regularizer;

    // Kernel parameters
    double kernel_neg_penalty, kernel_regularizer;

    // Sparse parameters
    double sparse_neg_penalty, sparse_regularizer;
    
    // Sequential parameters
    int sequential_dim;
    double sequential_neg_penalty, sequential_regularizer;

    // Common Neighbor parameters
    double normalizer;

    // Label Prediction parameters
    bool vec_normalize;
    double svm_regularizer;
    int svm_sample_ratio;

    // Link Prediction parameters
    double link_svm_regularizer;
    int link_svm_sample_ratio;

    // Predefined parameters
    std::string node_file, embedding_file;

    // SVD parameters
    std::string svd_u_file, svd_sv_file, svd_v_file;
};

void EvalFiniteEmbedding(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Finite Embedding\n";
    model.reset(GetFiniteEmbedding(config.train, config.neg_train, config.finite_dim, config.finite_neg_penalty, config.finite_regularizer));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.train, config.neg_train, config.test, config.neg_test);
        //std::cout << "Predicted AP: " << EvaluatePredictedAP(model.get(), config.train, config.test, config.neg_test, config.link_svm_regularizer, config.link_svm_sample_ratio) << "\n";
    }
    if (config.predict_label) {
        std::cout << "Evaluating Label Prediction\n";
        std::cout << "Average F1:" << EvaluateF1(model.get(), config.train_label, config.test_label, config.svm_regularizer, config.svm_sample_ratio, config.vec_normalize) << "\n";
        //std::cout << "Average F1 (Train):" << EvaluateF1(model.get(), config.train_label, config.train_label, config.svm_regularizer, config.svm_sample_ratio) << "\n";
    }
}

void EvalFiniteSGD(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Finite SGD\n";
    model.reset(GetFiniteSGD(config.train, config.neg_train, config.finite_dim, config.finite_neg_penalty, config.finite_regularizer));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.train, config.neg_train, config.test, config.neg_test);
        //std::cout << "Predicted AP: " << EvaluatePredictedAP(model.get(), config.train, config.test, config.neg_test, config.link_svm_regularizer, config.link_svm_sample_ratio) << "\n";
    }
    if (config.predict_label) {
        std::cout << "Evaluating Label Prediction\n";
        std::cout << "Average F1:" << EvaluateF1(model.get(), config.train_label, config.test_label, config.svm_regularizer, config.svm_sample_ratio, config.vec_normalize) << "\n";
        //std::cout << "Average F1 (Train):" << EvaluateF1(model.get(), config.train_label, config.train_label, config.svm_regularizer, config.svm_sample_ratio) << "\n";
    }
}

void EvalFiniteContrastEmbedding(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Finite Contrast Embedding\n";
    model.reset(GetFiniteContrastEmbedding(config.train, config.neg_train, config.finite_contrast_sample_ratio, config.finite_contrast_dim, config.finite_contrast_regularizer));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.train, config.neg_train, config.test, config.neg_test);
        //std::cout << "Predicted AP: " << EvaluatePredictedAP(model.get(), config.train, config.test, config.neg_test, config.link_svm_regularizer, config.link_svm_sample_ratio) << "\n";
    }
    if (config.predict_label) {
        std::cout << "Evaluating Label Prediction\n";
        std::cout << "Average F1:" << EvaluateF1(model.get(), config.train_label, config.test_label, config.svm_regularizer, config.svm_sample_ratio, config.vec_normalize) << "\n";
    }
}

void EvalDirectedFiniteEmbedding(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Directed Finite Embedding\n";
    model.reset(GetDirectedFiniteEmbedding(config.d_train, config.d_neg_train, config.d_finite_dim, config.d_finite_neg_penalty, config.d_finite_regularizer));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.d_train, config.d_neg_train, config.d_test, config.d_neg_test);
    }
    if (config.predict_label) {
        std::cout << "Evaluating Label Prediction\n";
        std::cout << "Average F1:" << EvaluateF1(model.get(), config.train_label, config.test_label, config.svm_regularizer, config.svm_sample_ratio, config.vec_normalize) << "\n";
        //std::cout << "Average F1 (Train):" << EvaluateF1(model.get(), config.train_label, config.train_label, config.svm_regularizer, config.svm_sample_ratio) << "\n";
    }
}

void EvalDirectedFiniteContrastEmbedding(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Finite Contrast Embedding\n";
    model.reset(GetDirectedFiniteContrastEmbedding(config.d_train, config.d_neg_train, config.d_finite_contrast_sample_ratio, config.d_finite_contrast_dim, config.d_finite_contrast_regularizer));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.d_train, config.d_neg_train, config.d_test, config.d_neg_test);
    }
    if (config.predict_label) {
        std::cout << "Evaluating Label Prediction\n";
        std::cout << "Average F1:" << EvaluateF1(model.get(), config.train_label, config.test_label, config.svm_regularizer, config.svm_sample_ratio, config.vec_normalize) << "\n";
    }
}

void EvalKernelEmbedding(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Kernel Embedding\n";
    model.reset(GetKernelEmbedding(config.train, config.neg_train, config.kernel_neg_penalty, config.kernel_regularizer));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        EvaluateAll(model.get(), config.train, config.neg_train, config.test, config.neg_test);
    }
}

void EvalSparseEmbedding(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    Graph neg_empty(config.train.size);
    std::cout << "Training Sparse Embedding\n";
    model.reset(GetSparseEmbedding(config.train, neg_empty, config.sparse_neg_penalty, config.sparse_regularizer));
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
        std::cout << EvaluateAveragePrecision(model.get(), config.test, config.neg_test) << "\n";
    }
}

void EvalPredefined(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training Predefined\n";
    model.reset(GetPredefined(config.node_file, config.embedding_file));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        std::cout << "Average Precision: " << EvaluateAveragePrecision(model.get(), config.test, config.neg_test) << "\n";
        std::cout << "Predicted Average Precision: " << EvaluatePredictedAP(model.get(), config.train, config.test, config.neg_test, config.link_svm_regularizer, config.link_svm_sample_ratio) << "\n";
    }
    if (config.predict_label) {
        std::cout << "Evaluating Label Prediction\n";
        std::cout << "Average F1:" << EvaluateF1(model.get(), config.train_label, config.test_label, config.svm_regularizer, config.svm_sample_ratio, config.vec_normalize) << "\n";
    }
}

void EvalSVD(const EvaluateConfig& config) {
    std::unique_ptr<Model> model;
    std::cout << "Training SVD\n";
    model.reset(GetSVD(config.node_file, config.svd_u_file, config.svd_sv_file, config.svd_v_file));
    if (config.predict_edge) {
        std::cout << "Evaluating Link Prediction\n";
        //std::cout << "Average Precision: " << EvaluateAveragePrecision(model.get(), config.test, config.neg_test) << "\n";
        std::cout << "Predicted Average Precision: " << EvaluatePredictedAP(model.get(), config.train, config.test, config.neg_test, config.link_svm_regularizer, config.link_svm_sample_ratio) << "\n";
    }
    if (config.predict_label) {
        std::cout << "Evaluating Label Prediction\n";
        std::cout << "Average F1:" << EvaluateF1(model.get(), config.train_label, config.test_label, config.svm_regularizer, config.svm_sample_ratio, config.vec_normalize) << "\n";
    }
}

void EvalLabelPropagation(const EvaluateConfig& config) {
    if (!config.predict_label) return;
    std::cout << "Training Label Propagation\n";
    std::cout << "Evaluating Label Prediction\n";
    std::cout << EvaluateF1LabelPropagation(config.train, config.train_label, config.test_label) << "\n";
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
    int test_case = 2;

    EvaluateConfig config;
    std::cout << "Reading Dataset\n";
    switch (test_case) {
    case 0:
        ReadDataset("node-w-filter.txt", "edge-ww-train-filter.txt", &config.train);
        ReadDataset("node-w-filter.txt", "edge-ww-val-filter.txt", &config.test);
        //ReadDataset("anonymized-tweet-node.txt", "anonymized-tweet-edge-train.txt", &config.train);
        //ReadDataset("anonymized-tweet-node.txt", "anonymized-tweet-edge-val.txt", &config.test);
        config.predict_edge = true;
        config.predict_label = false;

        config.finite_dim = 100; config.finite_neg_penalty = 0.03; config.finite_regularizer = 5;
        config.finite_contrast_sample_ratio = 6; config.finite_contrast_dim = 100; config.finite_contrast_regularizer = 120;
        config.kernel_neg_penalty = 0.03; config.kernel_regularizer = 5;
        config.sparse_neg_penalty = 0.015; config.sparse_regularizer = 15;
        config.sequential_dim = 100; config.sequential_neg_penalty = 0.1; config.sequential_regularizer = 2;
        config.link_svm_regularizer = 1; config.link_svm_sample_ratio = 3;
        config.normalizer = 120;
        config.node_file = "node-w-filter.txt"; config.embedding_file = "line-tweet-vec.txt";
        //config.node_file = "anonymized-tweet-node.txt"; config.embedding_file = "n2vtweet-embedding.txt";
        config.svd_u_file = "tweet-svd-u.txt"; config.svd_sv_file = "tweet-svd-sigma.txt"; config.svd_v_file = "tweet-svd-v.txt";
        break;
    case 1:
        ReadDataset("blog-node.txt", "blog-edge-train.txt", &config.train);
        ReadDataset("blog-node.txt", "blog-edge-val.txt", &config.test);
        ReadLabel("blog-node.txt", "blog-label-train.txt", &config.train_label);
        ReadLabel("blog-node.txt", "blog-label-val.txt", &config.test_label);
        config.predict_edge = true;
        config.predict_label = true;

        config.finite_dim = 100; config.finite_neg_penalty = 0.03; config.finite_regularizer = 3;
        config.finite_contrast_sample_ratio = 4; config.finite_contrast_dim = 100; config.finite_contrast_regularizer = 55;
        config.kernel_neg_penalty = 0.03; config.kernel_regularizer = 3;
        config.sparse_neg_penalty = 0.015; config.sparse_regularizer = 15;
        config.sequential_dim = 100; config.sequential_neg_penalty = 0.1; config.sequential_regularizer = 2;
        config.svm_regularizer = 1; config.svm_sample_ratio = 5; config.vec_normalize = true;
        config.link_svm_regularizer = 1; config.link_svm_sample_ratio = 3;
        config.normalizer = 120;
        config.node_file = "blog-node.txt"; config.embedding_file = "n2vblog-embedding.txt";
        config.svd_u_file = "blog-svd-u.txt"; config.svd_sv_file = "blog-svd-sigma.txt"; config.svd_v_file = "blog-svd-v.txt";
        break;
    case 2:
        ReadDataset("youtube-node.txt", "youtube-edge-undirected-train.txt", &config.train);
        ReadDataset("youtube-node.txt", "youtube-edge-undirected-val.txt", &config.test);
        //ReadDirectedDataset("youtube-node.txt", "youtube-edge-train.txt", &config.d_train);
        //ReadDirectedDataset("youtube-node.txt", "youtube-edge-val.txt", &config.d_test);
        ReadLabel("youtube-node.txt", "youtube-label-train.txt", &config.train_label);
        ReadLabel("youtube-node.txt", "youtube-label-val.txt", &config.test_label);
        config.predict_edge = true;
        config.predict_label = true;

        config.finite_dim = 100; config.finite_neg_penalty = 0.03; config.finite_regularizer = 1;
        config.finite_contrast_sample_ratio = 4; config.finite_contrast_dim = 100; config.finite_contrast_regularizer = 30;
        config.d_finite_dim = 100; config.d_finite_neg_penalty = 0.03; config.d_finite_regularizer = 5;
        config.d_finite_contrast_sample_ratio = 4; config.d_finite_contrast_dim = 100; config.d_finite_contrast_regularizer = 50;
        config.kernel_neg_penalty = 0.03; config.kernel_regularizer = 30;
        config.sparse_neg_penalty = 0.015; config.sparse_regularizer = 15;
        config.sequential_dim = 100; config.sequential_neg_penalty = 0.1; config.sequential_regularizer = 2;
        config.svm_regularizer = 1; config.svm_sample_ratio = 5; config.vec_normalize = true;
        config.link_svm_regularizer = 1; config.link_svm_sample_ratio = 2;
        config.normalizer = 120;
        config.node_file = "youtube-node.txt"; config.embedding_file = "line-youtube-vec.txt";
        config.svd_u_file = "youtube-svd-u.txt"; config.svd_sv_file = "youtube-svd-sigma.txt"; config.svd_v_file = "youtube-svd-v.txt";
        break;
    }

    std::cout << "Sampling Negative Dataset\n";
    config.neg_train = Graph(config.train.size);
    SampleNegativeGraphUniform(config.train, &config.neg_train);
    
    config.neg_test = Graph(config.test.size);
    //SampleNegativeGraphPreferential(test, &neg_test, 1);    
    SampleNegativeGraphUniform(config.test, &config.neg_test);
    
    RemoveRedundant(config.train, &config.neg_train);
    RemoveRedundant(config.train, &config.neg_test);
    RemoveRedundant(config.test, &config.neg_test);

    //EvalFiniteEmbedding(config);
    EvalFiniteSGD(config);
    //EvalFiniteContrastEmbedding(config);
    //EvalKernelEmbedding(config);
    //EvalSparseEmbedding(train, neg_empty, test, neg_test);
    //EvalSequentialEmbedding(train, neg_train, test, neg_test);
    //EvalCommonNeighbor(config);
    //EvalPredefined(config);
    //EvalLabelPropagation(config);
    //EvalRandom(config);
    //EvalSVD(config);

    system("pause");
}