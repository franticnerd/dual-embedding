#include "base.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>

void ReadDataset(const std::string& nodefile, const std::string& edgefile, Graph* graph) {
    std::map<std::string, int> index_map;
    char buffer[256];
    int index = 0;

    std::ifstream fin(nodefile);
    while (fin.getline(buffer, 256))
        index_map[std::string(buffer)] = index++;

    std::ifstream fin2(edgefile);
    char w1[256], w2[256];
    double weight;
    *graph = Graph(index);

    int e_index = 0;    
    while (fin2.getline(buffer, 256)) {
        std::istringstream is(buffer);
        is.getline(w1, 256, '\t');
        is.getline(w2, 256, '\t');
        is >> weight;
        int w1_index = index_map[std::string(w1)];
        int w2_index = index_map[std::string(w2)];
        if (w1_index < w2_index) {
            graph->AddEdge(w1_index, w2_index);
            ++e_index;
        }
    }

    std::cout << "Node File: " << nodefile << "; Edge File: " << edgefile << "; Total Edges: " << e_index << "\n";
}

void ReadDirectedDataset(const std::string& nodefile, const std::string& edgefile, DGraph* graph) {
    std::map<std::string, int> index_map;
    char buffer[256];
    int index = 0;

    std::ifstream fin(nodefile);
    while (fin.getline(buffer, 256))
        index_map[std::string(buffer)] = index++;

    std::ifstream fin2(edgefile);
    char w1[256], w2[256];
    double weight;
    *graph = DGraph(index);

    int e_index = 0;
    while (fin2.getline(buffer, 256)) {
        std::istringstream is(buffer);
        is.getline(w1, 256, '\t');
        is.getline(w2, 256, '\t');
        is >> weight;
        int w1_index = index_map[std::string(w1)];
        int w2_index = index_map[std::string(w2)];
        if (w1_index < w2_index) {
            graph->AddEdge(w1_index, w2_index);
            ++e_index;
        }
    }

    std::cout << "Node File: " << nodefile << "; Edge File: " << edgefile << "; Total Edges: " << e_index << "\n";
}

void ReadLabel(const std::string& nodefile, const std::string& labelfile, Label* label) {
    std::map<std::string, int> index_map;
    char buffer[256];
    int index = 0;

    std::ifstream fin(nodefile);
    while (fin.getline(buffer, 256))
        index_map[std::string(buffer)] = index++;

    std::ifstream fin2(labelfile);
    char w1[256], w2[256];
    *label = Label(index);

    int e_index = 0;
    while (fin2.getline(buffer, 256)) {
        std::istringstream is(buffer);
        is.getline(w1, 256, '\t');
        is >> w2;
        int w1_index = index_map[std::string(w1)];
        int w2_index = std::stoi(std::string(w2));
        label->SetLabel(w1_index, w2_index - 1);
    }

    std::cout << "Node File: " << nodefile << "; Label File: " << labelfile << "; Total Labels: " << label->card << "\n";
}