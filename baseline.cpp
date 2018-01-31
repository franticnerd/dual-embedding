#include "base.h"
#include "utility.h"

#include <set>
#include <vector>
#include <cmath>
#include <random>
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>

namespace {
    std::mt19937 gen;
}   // anonymous namespace

class CommonNeighbor : public Model {
    Graph base_;
    const double normalizer_;
  public:
    CommonNeighbor(const Graph& base, double normalizer) : base_(base), normalizer_(normalizer) {}
    double Evaluate(int x, int y) {
        std::set<int> set;
        for (const auto& p : base_.edge[x])
            set.insert(p);
        double val = 0;
        for (const auto& p : base_.edge[y])
            if (set.count(p) > 0)
                val++;
        if (set.count(y) > 0) 
            val += sqrt(base_.edge[x].size()) + sqrt(base_.edge[y].size());
        return val / normalizer_;
    }
};

class AdamicAdar : public Model {
    Graph base_;
    std::vector<int> cnt_;
  public:
    AdamicAdar(const Graph& base) : base_(base), cnt_(base.size, 0) {}
    double Evaluate(int x, int y) {
        for (int p : base_.edge[x])
            cnt_[p] = 1;
        double val = 0;
        for (int p : base_.edge[y])
            if (cnt_[p] == 1)
                val += 1 / log(base_.edge[p].size());
        for (int p : base_.edge[x])
            cnt_[p] = 0;
        return val;
    }
};

class Random : public Model {
public:
    Random() {}
    double Evaluate(int x, int y) {
        std::uniform_real_distribution<double> dist(0, 1);
        return dist(gen);
    }
};

class Predefined : public Model {
    int n_, dim_;
    std::vector<std::vector<double>> embedding;
public:
    Predefined(const std::string& node_file, const std::string& embedding_file) {
        std::map<std::string, int> index_map;
        char buffer[2500];
        int index = 0;

        std::ifstream fin(node_file);
        while (fin.getline(buffer, 256))
            index_map[std::string(buffer)] = index++;

        std::ifstream fin2(embedding_file);
        fin2.getline(buffer, 2500);
        std::istringstream is(buffer);
        is >> n_ >> dim_;

        embedding.resize(index);
        for (int i = 0; i < index; ++i)
            embedding[i].resize(dim_, 0);

        for (int i = 0; i < n_; ++i) {
            fin2.getline(buffer, 2500);

            std::istringstream is2(buffer);
            std::string word;
            std::vector<std::string> word_vec;
            while (is2 >> word)
                word_vec.push_back(word);
            
            word = "";
            for (int j = 0; j < (int)word_vec.size() - dim_; ++j) {
                if (j != 0) word.append(" ");
                word.append(word_vec[j]);                
            }
            int node_index = index_map.at(word);

            for (int j = word_vec.size() - dim_; j < (int)word_vec.size(); ++j)
                embedding[node_index][j - (word_vec.size() - dim_)] = std::stof(word_vec[j]);
        }
    }
    double Evaluate(int x, int y) {
        return InnerProduct(embedding[x].data(), embedding[y].data(), dim_);
    }
    const std::vector<double>& GetEmbedding(int x) { return embedding[x]; }
};

class SVD : public Model {
    int n_, dim_;
    std::vector<std::vector<double>> u_;
    std::vector<double> sv_;

    void ReadVec(const std::map<std::string, int>& index_map, std::vector<std::vector<double>>* vec, const std::string& vec_file) {
        char buffer[2500];
        std::ifstream fin(vec_file);
        fin.getline(buffer, 2500);
        std::istringstream is(buffer);
        is >> n_ >> dim_;

        vec->resize(n_);
        for (int i = 0; i < n_; ++i)
            vec->at(i).resize(dim_, 0);

        for (int i = 0; i < n_; ++i) {
            fin.getline(buffer, 2500);

            std::istringstream is2(buffer);
            std::string word;
            std::vector<std::string> word_vec;
            while (is2 >> word)
                word_vec.push_back(word);

            word = "";
            for (int j = 0; j < (int)word_vec.size() - dim_; ++j) {
                if (j != 0) word.append(" ");
                word.append(word_vec[j]);
            }
            int node_index = index_map.at(word);

            for (int j = word_vec.size() - dim_; j < (int)word_vec.size(); ++j) {
                vec->at(node_index)[j - (word_vec.size() - dim_)] = std::stod(word_vec[j]);
            }
        }

    }
public:
    SVD(const std::string& node_file, const std::string& u_file, const std::string& sv_file, const std::string& v_file) {
        std::map<std::string, int> index_map;
        char buffer[2500];
        int index = 0;

        std::ifstream fin(node_file);
        while (fin.getline(buffer, 256))
            index_map[std::string(buffer)] = index++;

        ReadVec(index_map, &u_, u_file);
        //ReadVec(index_map, &v_, v_file);

        std::ifstream fin2(sv_file);
        sv_.resize(dim_);
        for (int i = 0; i < dim_; ++i)
            fin2 >> sv_[i];
    }
    double Evaluate(int x, int y) {
        double val = 0;
        for (int i = 0; i < dim_; ++i)
            val += u_[x][i] * u_[x][i] * sv_[i];
        return val;
    }
    const std::vector<double>& GetEmbedding(int x) { return u_[x]; }
};

Model* GetCommonNeighbor(const Graph& base, double normalizer) {
    return new CommonNeighbor(base, normalizer);
}

Model* GetAdamicAdar(const Graph& base) {
    return new AdamicAdar(base);
}

Model* GetPredefined(const std::string& node_file, const std::string& embedding_file) {
    return new Predefined(node_file, embedding_file);
}

Model* GetRandom() {
    return new Random();
}

Model* GetSVD(const std::string& node_file, const std::string& u_file, const std::string& sv_file, const std::string& v_file) {
    return new SVD(node_file, u_file, sv_file, v_file);
}