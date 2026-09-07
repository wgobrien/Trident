#ifndef COVERAGE_H
#define COVERAGE_H

#include <string>
#include <unordered_set>
#include <fstream>
#include <iostream>

inline void get_read_ids(
    std::unordered_set<std::string>& read_ids,
    const std::string& kraken_file
) {
    if (kraken_file.empty())
        return;

    std::ifstream in(kraken_file);
    if (!in) {
        std::cerr << "Failed to open Kraken file: " << kraken_file << "\n";
        exit(1);
    }

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty())
            continue;

        size_t t1 = line.find('\t');
        if (t1 == std::string::npos)
            continue;

        size_t t2 = line.find('\t', t1 + 1);
        if (t2 == std::string::npos)
            continue;

        read_ids.insert(line.substr(t1 + 1, t2 - t1 - 1));
    }
}

#endif
