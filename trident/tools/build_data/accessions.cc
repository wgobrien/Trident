#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>
#include <filesystem>
#include <cstdlib>
#include <zlib.h>

std::unordered_map<std::string, int>
build_a2t(const std::string& gz_path,
          const std::unordered_set<std::string>& accessions)
{
    const std::string url =
        "https://ftp.ncbi.nih.gov/pub/taxonomy/accession2taxid/nucl_gb.accession2taxid.gz";

    if (!std::filesystem::exists(gz_path)) {
        std::filesystem::create_directories(std::filesystem::path(gz_path).parent_path());
        std::string cmd = "curl -L -o " + gz_path + " " + url;
        if (std::system(cmd.c_str()) != 0) throw std::runtime_error("download failed");
    }

    gzFile gz = gzopen(gz_path.c_str(), "rb");
    if (!gz) throw std::runtime_error("open failed");

    char buf[1 << 16];
    if (!gzgets(gz, buf, sizeof(buf))) throw std::runtime_error("header read failed");

    std::string header_line(buf);
    header_line.pop_back();

    int acc_idx = -1, taxid_idx = -1, col = 0;
    size_t pos = 0;

    while (true) {
        size_t next = header_line.find('\t', pos);
        std::string_view field(header_line.data() + pos,
                               (next == std::string::npos ? header_line.size() : next) - pos);
        if (field == "accession.version") acc_idx = col;
        if (field == "taxid") taxid_idx = col;
        if (next == std::string::npos) break;
        pos = next + 1;
        ++col;
    }

    if (acc_idx < 0 || taxid_idx < 0) throw std::runtime_error("missing columns");

    std::unordered_map<std::string, int> a2t;
    a2t.reserve(accessions.size());

    while (gzgets(gz, buf, sizeof(buf))) {
        std::string_view line(buf);
        if (!line.empty() && line.back() == '\n') line.remove_suffix(1);

        int cur_col = 0;
        std::string_view acc;
        std::string_view tax;

        size_t p = 0;
        while (true) {
            size_t n = line.find('\t', p);
            std::string_view field(line.data() + p,
                                   (n == std::string::npos ? line.size() : n) - p);

            if (cur_col == acc_idx) acc = field;
            else if (cur_col == taxid_idx) tax = field;

            if (n == std::string::npos) break;
            p = n + 1;
            ++cur_col;
        }

        if (!acc.empty() && accessions.count(std::string(acc))) {
            a2t.emplace(std::string(acc), std::stoi(std::string(tax)));
            if (a2t.size() == accessions.size()) break;
        }
    }

    gzclose(gz);
    return a2t;
}

// int main(int argc, char* argv[])
// {
//     if (argc < 3) {
//         std::cerr << "usage: " << argv[0]
//                   << " <a2t_path> <accession1> [accession2 ...]\n";
//         return 1;
//     }

//     std::string a2t_path = argv[1];
//     std::unordered_set<std::string> accessions;
//     for (int i = 2; i < argc; ++i) {
//         accessions.insert(argv[i]);
//     }

//     auto result = build_a2t(a2t_path, accessions);

//     for (const auto& [acc, taxid] : result) {
//         std::cout << acc << "\t" << taxid << "\n";
//     }

//     return 0;
// }
