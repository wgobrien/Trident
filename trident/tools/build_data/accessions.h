#ifndef BUILD_A2T_H
#define BUILD_A2T_H

#include <string>
#include <unordered_map>
#include <unordered_set>

std::unordered_map<std::string, int>
build_a2t(const std::string& a2t_path,
          const std::unordered_set<std::string>& accessions);

#endif