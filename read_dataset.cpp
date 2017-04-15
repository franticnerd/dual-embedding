#include "base.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <map>

void ReadDataset(const std::string& nodefile, const std::string& edgefile, Graph* graph) {
    std::ifstream fin(nodefile);
    std::map<std::string, int> index_map;
    char buffer[256];
    int index = 0;
    while (fin.getline(buffer, 256))
        index_map[std::string(buffer)] = index++;

    std::ifstream fin2(edgefile);
    char w1[256], w2[256];
    double weight;
    *graph = Graph(index);
    while (fin2.getline(buffer, 256)) {
        std::istringstream is(buffer);
        is.getline(w1, 256, '\t');
        is.getline(w2, 256, '\t');
        is >> weight;
        int w1_index = index_map[std::string(w1)];
        int w2_index = index_map[std::string(w2)];
        graph->AddEdge(w1_index, w2_index);
    }
}