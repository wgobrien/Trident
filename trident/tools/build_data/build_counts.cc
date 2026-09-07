#include "mmscanner.h"
#include "kraken2_headers.h"
#include "coverage_headers.h"
#include "coverage.h"

#include <algorithm>

using namespace kraken2;
using std::cerr;
using std::cout;
using std::string;
using std::ifstream;
using std::getline;
using std::unordered_set;
using std::unordered_map;
using std::regex;
using std::regex_search;
using std::smatch;
using std::vector;
using std::thread;
using std::mutex;
using std::condition_variable;
using std::queue;
using std::unique_lock;
using std::lock_guard;
using std::pair;

struct FlatMinimizerMap {
    vector<uint64_t> keys;   // minimizer hashes, sorted
    vector<uint32_t> vals;   // accession indices, parallel to keys

    void build(vector<pair<uint64_t, uint32_t>>& raw, uint32_t max_acc_per_mm = 0) {
        std::sort(raw.begin(), raw.end());
        raw.erase(std::unique(raw.begin(), raw.end()), raw.end());

        if (max_acc_per_mm > 0 && !raw.empty()) {
            size_t before = raw.size();
            size_t write = 0;
            size_t i = 0;
            size_t pruned_mm = 0;
            while (i < raw.size()) {
                size_t j = i;
                while (j < raw.size() && raw[j].first == raw[i].first)
                    ++j;
                if (static_cast<uint32_t>(j - i) <= max_acc_per_mm) {
                    for (size_t k = i; k < j; ++k)
                        raw[write++] = raw[k];
                } else {
                    ++pruned_mm;
                }
                i = j;
            }
            raw.resize(write);
            raw.shrink_to_fit();
            cerr << "[INFO] Minimizer filter: " << before << " -> "
                 << raw.size() << " entries (" << pruned_mm
                 << " over-represented minimizers removed, cap=" << max_acc_per_mm << ")\n";
        }

        // Structure-of-arrays: 12 bytes/entry vs 16 bytes/entry with pair
        keys.resize(raw.size());
        vals.resize(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) {
            keys[i] = raw[i].first;
            vals[i] = raw[i].second;
        }
        keys.shrink_to_fit();
        vals.shrink_to_fit();
        raw.clear();
        raw.shrink_to_fit();
    }

    pair<size_t, size_t> lookup(uint64_t mm) const {
        auto lo = std::lower_bound(keys.begin(), keys.end(), mm);
        auto hi = std::upper_bound(lo, keys.end(), mm);
        return {static_cast<size_t>(lo - keys.begin()),
                static_cast<size_t>(hi - keys.begin())};
    }

    size_t size() const { return keys.size(); }
    size_t memory_bytes() const {
        return keys.capacity() * sizeof(uint64_t) + vals.capacity() * sizeof(uint32_t);
    }
};

struct Reference {
    string header;
    string seq;
};

struct ScannerParams {
    ssize_t k;
    ssize_t l;
    uint64_t spaced_seed_mask;
    bool dna;
    uint64_t toggle_mask;
    int revcom_version;
};

struct ReadRecord {
    string id;
    string seq;
};

struct ReadResult {
    string read_id;
    unordered_map<uint32_t, uint32_t> acc_counts;
};

unordered_set<uint64_t> get_kmer_set(
    const string& seq,
    const ScannerParams& params
) {
    MinimizerScanner scanner(
        params.k,
        params.l,
        params.spaced_seed_mask,
        params.dna,
        params.toggle_mask,
        params.revcom_version
    );

    unordered_set<uint64_t> kmer_set;
    scanner.LoadSequence(seq, 0, seq.size());

    uint64_t* m;
    while ((m = scanner.NextMinimizer()) != nullptr)
        kmer_set.insert(*m);

    return kmer_set;
}

vector<uint64_t> get_minimizer_sequence(
    const string& seq,
    const ScannerParams& params
) {
    MinimizerScanner scanner(
        params.k,
        params.l,
        params.spaced_seed_mask,
        params.dna,
        params.toggle_mask,
        params.revcom_version
    );

    vector<uint64_t> minimizers;
    scanner.LoadSequence(seq, 0, seq.size());

    uint64_t* m;
    while ((m = scanner.NextMinimizer()) != nullptr)
        minimizers.push_back(*m);

    return minimizers;
}

unordered_map<uint32_t, uint32_t> count_acc_hits(
    const vector<uint64_t>& minimizers,
    const FlatMinimizerMap& mm_to_acc
) {
    unordered_map<uint32_t, uint32_t> counts;
    for (auto mm : minimizers) {
        auto [lo, hi] = mm_to_acc.lookup(mm);
        for (size_t i = lo; i < hi; ++i)
            counts[mm_to_acc.vals[i]]++;
    }
    return counts;
}

unordered_set<string> load_accessions(const string& path) {
    ifstream in(path);
    if (!in) {
        cerr << "Failed to open accessions file: " << path << "\n";
        exit(1);
    }

    unordered_set<string> accessions;
    string line;
    while (getline(in, line)) {
        if (line.empty()) continue;
        accessions.insert(line);
    }
    return accessions;
}

bool read_fasta_record(
    std::istream& in,
    Reference& rec,
    std::string& next_header
) {
    rec.header.clear();
    rec.seq.clear();

    std::string line;

    if (!next_header.empty()) {
        rec.header = std::move(next_header);
        next_header.clear();
    } else {
        while (std::getline(in, line)) {
            if (!line.empty() && line[0] == '>') {
                rec.header = line.substr(1);
                break;
            }
        }
    }

    if (rec.header.empty())
        return false;

    while (std::getline(in, line)) {
        if (!line.empty() && line[0] == '>') {
            next_header = line.substr(1);
            break;
        }
        rec.seq += line;
    }

    return true;
}

string extract_accession(const string& header) {
    static const regex acc_re("[A-Z]{1,2}_?\\d+\\.\\d+");
    smatch m;
    if (regex_search(header, m, acc_re))
        return m[0].str();
    size_t end = header.find_first_of(" \t");
    return header.substr(0, end);
}

void build_minimizer_map(
    const string& library_path,
    const unordered_set<string>& valid_accessions,
    ScannerParams& params,
    FlatMinimizerMap& mm_to_acc,
    vector<string>& acc_list,
    unordered_map<string, uint32_t>& acc_to_idx,
    int threads,
    uint32_t max_acc_per_mm
) {
    ifstream in(library_path);
    if (!in) {
        cerr << "Failed to open library file\n";
        exit(1);
    }

    mutex in_mutex;
    mutex idx_mutex;
    string next_header;
    std::atomic<size_t> processed{0};

    // Each thread collects (minimizer, acc_idx) pairs into a local buffer.
    vector<vector<pair<uint64_t, uint32_t>>> thread_buffers(threads);

    auto worker = [&](int tid) {
        Reference rec;
        auto& local_buf = thread_buffers[tid];
        while (true) {
            {
                lock_guard<mutex> lock(in_mutex);
                if (!read_fasta_record(in, rec, next_header))
                    return;
            }

            string acc = extract_accession(rec.header);
            if (!valid_accessions.count(acc))
                continue;

            uint32_t idx;
            {
                lock_guard<mutex> lock(idx_mutex);
                auto it = acc_to_idx.find(acc);
                if (it == acc_to_idx.end()) {
                    idx = acc_list.size();
                    acc_list.push_back(acc);
                    acc_to_idx[acc] = idx;
                } else {
                    idx = it->second;
                }
            }

            auto kmers = get_kmer_set(rec.seq, params);
            for (auto m : kmers) {
                local_buf.emplace_back(m, idx);
            }

            size_t c = ++processed;
            if (c % 100 == 0) {
                cerr << "\r[PROGRESS] references indexed: " << c << std::flush;
            }
        }
    };

    vector<thread> pool;
    for (int i = 0; i < threads; ++i)
        pool.emplace_back(worker, i);

    for (auto& t : pool)
        t.join();

    cerr << "\n";

    // Merge all thread-local buffers into one flat array.
    size_t total = 0;
    for (auto& buf : thread_buffers) total += buf.size();

    vector<pair<uint64_t, uint32_t>> raw;
    raw.reserve(total);
    for (auto& buf : thread_buffers) {
        raw.insert(raw.end(),
                   std::make_move_iterator(buf.begin()),
                   std::make_move_iterator(buf.end()));
        buf.clear();
        buf.shrink_to_fit();
    }
    thread_buffers.clear();

    cerr << "[INFO] Sorting & deduplicating " << raw.size() << " minimizer entries...\n";
    mm_to_acc.build(raw, max_acc_per_mm);
}

struct ReadPair {
    string id;
    string seq1;
    string seq2;  // empty if unpaired
};

bool read_next_record(gzFile fp, char* buffer, size_t buf_size, string& read_id, string& seq) {
    string line;
    read_id.clear();
    seq.clear();

    while (gzgets(fp, buffer, buf_size)) {
        line.assign(buffer);
        if (!line.empty() && line.back() == '\n')
            line.pop_back();
        if (line.empty()) continue;

        if (line[0] == '>' || line[0] == '@') {
            size_t end = line.find_first_of(" \t", 1);
            read_id = line.substr(1, end == string::npos ? string::npos : end - 1);

            while (gzgets(fp, buffer, buf_size)) {
                line.assign(buffer);
                if (!line.empty() && line.back() == '\n')
                    line.pop_back();
                if (line.empty()) continue;

                if (line[0] == '>' || line[0] == '@') {
                    break;
                } else if (line[0] == '+') {
                    size_t qlen = 0;
                    while (qlen < seq.size() && gzgets(fp, buffer, buf_size)) {
                        qlen += std::strlen(buffer) - (buffer[strlen(buffer)-1] == '\n' ? 1 : 0);
                    }
                    break;
                } else {
                    seq += line;
                }
            }
            return true;
        }
    }
    return false;
}

string strip_pair_suffix(const string& id) {
    if (id.size() >= 2 && id[id.size()-2] == '/') {
        char last = id.back();
        if (last == '1' || last == '2') {
            return id.substr(0, id.size() - 2);
        }
    }
    return id;
}

// Streaming read processor: processes reads in batches and writes COO triplets
// to temp files, so memory usage is O(batch_size) not O(total_reads).
struct StreamingCOOWriter {
    std::ofstream row_file, col_file, data_file, row_ids_file;
    int64_t n_reads = 0;
    int64_t nnz = 0;

    void open(const string& tmp_prefix, const string& row_ids_path) {
        row_file.open(tmp_prefix + ".rows.tmp", std::ios::binary);
        col_file.open(tmp_prefix + ".cols.tmp", std::ios::binary);
        data_file.open(tmp_prefix + ".data.tmp", std::ios::binary);
        row_ids_file.open(row_ids_path);
        if (!row_file || !col_file || !data_file || !row_ids_file) {
            cerr << "Failed to open temp files for streaming output\n";
            exit(1);
        }
    }

    void write_batch(const vector<ReadResult>& batch) {
        for (const auto& r : batch) {
            uint32_t row = static_cast<uint32_t>(n_reads);
            row_ids_file << r.read_id << '\n';
            for (const auto& [col, val] : r.acc_counts) {
                row_file.write(reinterpret_cast<const char*>(&row), 4);
                col_file.write(reinterpret_cast<const char*>(&col), 4);
                data_file.write(reinterpret_cast<const char*>(&val), 4);
                ++nnz;
            }
            ++n_reads;
        }
    }

    void close() {
        row_file.close();
        col_file.close();
        data_file.close();
        row_ids_file.close();
    }

    // Assemble final binary from the 3 temp files and clean up.
    void assemble(const string& out_path, const string& tmp_prefix, int64_t n_accs) {
        close();

        std::ofstream out(out_path, std::ios::binary);
        if (!out) {
            cerr << "Failed to open output file: " << out_path << "\n";
            exit(1);
        }

        out.write("TCOO", 4);
        out.write(reinterpret_cast<const char*>(&n_reads), 8);
        out.write(reinterpret_cast<const char*>(&n_accs), 8);
        out.write(reinterpret_cast<const char*>(&nnz), 8);

        // Append each temp file to the output.
        auto append_tmp = [&](const string& path) {
            std::ifstream tmp(path, std::ios::binary);
            if (!tmp) {
                cerr << "Failed to read temp file: " << path << "\n";
                exit(1);
            }
            constexpr size_t CHUNK = 1 << 20;  // 1 MB
            vector<char> buf(CHUNK);
            while (tmp.read(buf.data(), CHUNK) || tmp.gcount() > 0) {
                out.write(buf.data(), tmp.gcount());
            }
            tmp.close();
            std::remove(path.c_str());
        };

        append_tmp(tmp_prefix + ".rows.tmp");
        append_tmp(tmp_prefix + ".cols.tmp");
        append_tmp(tmp_prefix + ".data.tmp");

        out.close();
    }
};

void process_reads_streaming(
    const string& reads_path,
    const string& paired_reads_path,
    ScannerParams& params,
    const FlatMinimizerMap& mm_to_acc,
    const unordered_set<string>& allowed_read_ids,
    int threads,
    StreamingCOOWriter& writer,
    size_t batch_size = 10000
) {
    constexpr size_t BUF_SIZE = 1 << 20;
    char buffer1[BUF_SIZE];
    char buffer2[BUF_SIZE];

    gzFile fp1 = gzopen(reads_path.c_str(), "rb");
    if (!fp1) {
        cerr << "Failed to open reads file: " << reads_path << "\n";
        exit(1);
    }

    gzFile fp2 = nullptr;
    if (!paired_reads_path.empty()) {
        fp2 = gzopen(paired_reads_path.c_str(), "rb");
        if (!fp2) {
            cerr << "Failed to open paired reads file: " << paired_reads_path << "\n";
            gzclose(fp1);
            exit(1);
        }
    }

    size_t total_reads = 0;
    string id1, seq1, id2, seq2;

    // Read and process in batches.
    while (true) {
        // Fill a batch of read pairs.
        vector<ReadPair> batch;
        batch.reserve(batch_size);

        while (batch.size() < batch_size && read_next_record(fp1, buffer1, BUF_SIZE, id1, seq1)) {
            ReadPair pair;
            pair.id = strip_pair_suffix(id1);

            if (!allowed_read_ids.empty() && !allowed_read_ids.count(pair.id)) {
                if (fp2) read_next_record(fp2, buffer2, BUF_SIZE, id2, seq2);
                continue;
            }

            pair.seq1 = std::move(seq1);

            if (fp2 && read_next_record(fp2, buffer2, BUF_SIZE, id2, seq2)) {
                string stripped2 = strip_pair_suffix(id2);
                if (pair.id != stripped2) {
                    cerr << "[WARN] Read pair mismatch: " << pair.id << " vs " << stripped2 << "\n";
                }
                pair.seq2 = std::move(seq2);
            }

            batch.push_back(std::move(pair));
        }

        if (batch.empty()) break;

        // Process batch with thread pool.
        vector<ReadResult> results(batch.size());
        std::atomic<size_t> next_idx{0};

        auto worker = [&]() {
            while (true) {
                size_t i = next_idx.fetch_add(1);
                if (i >= batch.size()) return;

                auto& rec = batch[i];
                auto counts = count_acc_hits(get_minimizer_sequence(rec.seq1, params), mm_to_acc);

                if (!rec.seq2.empty()) {
                    auto counts2 = count_acc_hits(get_minimizer_sequence(rec.seq2, params), mm_to_acc);
                    for (auto& [acc_idx, cnt] : counts2)
                        counts[acc_idx] += cnt;
                }

                if (!counts.empty()) {
                    results[i] = {std::move(rec.id), std::move(counts)};
                }
            }
        };

        vector<thread> workers;
        for (int t = 0; t < threads; ++t)
            workers.emplace_back(worker);
        for (auto& t : workers)
            t.join();

        // Remove empty results (reads with no hits) and write batch.
        vector<ReadResult> hits;
        hits.reserve(batch.size());
        for (auto& r : results) {
            if (!r.acc_counts.empty())
                hits.push_back(std::move(r));
        }

        writer.write_batch(hits);
        total_reads += batch.size();

        cerr << "\r[PROGRESS] read pairs processed: " << total_reads << std::flush;
    }

    gzclose(fp1);
    if (fp2) gzclose(fp2);

    cerr << "\n[READS] " << total_reads << " total read pairs processed, "
         << writer.n_reads << " with hits, "
         << writer.nnz << " nonzero entries\n";
}

void write_col_ids(const vector<string>& acc_list, const string& out_path) {
    std::ofstream col_ids_out(out_path + ".col_ids");
    if (!col_ids_out) {
        cerr << "Failed to open col_ids output file\n";
        exit(1);
    }
    for (const auto& acc : acc_list)
        col_ids_out << acc << '\n';
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: ./build_counts <reads> <library> <accessions_file> "
                "[--pair paired_reads] "
                "[--output FILE] "
                "[--kraken-file FILE] "
                "[--threads N] "
                "[--k K] "
                "[--l L] "
                "[--spaces N] "
                "[--max-acc-per-mm N]\n";
        return 1;
    }

    string reads = argv[1];
    string library = argv[2];
    string accessions_file = argv[3];

    string paired_reads;
    string out_path = "counts.bin";
    string kraken_file;

    int threads = 1;
    ssize_t k = 35;
    ssize_t l = 31;
    uint64_t spaces = 7;
    uint32_t max_acc_per_mm = 512;

    for (int i = 4; i < argc; ++i) {
        string arg = argv[i];

        if (arg == "--pair") {
            if (i + 1 >= argc) {
                cerr << "--pair requires an argument\n";
                return 1;
            }
            paired_reads = argv[++i];

        } else if (arg == "--kraken-file") {
            if (i + 1 >= argc) {
                cerr << "--kraken-file requires an argument\n";
                return 1;
            }
            kraken_file = argv[++i];

        } else if (arg == "--output") {
            if (i + 1 >= argc) {
                cerr << "--output requires an argument\n";
                return 1;
            }
            out_path = argv[++i];

        } else if (arg == "--threads") {
            if (i + 1 >= argc) {
                cerr << "--threads requires an argument\n";
                return 1;
            }
            threads = std::stoi(argv[++i]);
            if (threads < 1) {
                cerr << "--threads must be >= 1\n";
                return 1;
            }

        } else if (arg == "--k") {
            if (i + 1 >= argc) {
                cerr << "--k requires an argument\n";
                return 1;
            }
            k = std::stoll(argv[++i]);
            if (k < 1) {
                cerr << "--k must be >= 1\n";
                return 1;
            }

        } else if (arg == "--l") {
            if (i + 1 >= argc) {
                cerr << "--l requires an argument\n";
                return 1;
            }
            l = std::stoll(argv[++i]);
            if (l < 1) {
                cerr << "--l must be >= 1\n";
                return 1;
            }

        } else if (arg == "--spaces") {
            if (i + 1 >= argc) {
                cerr << "--spaces requires an argument\n";
                return 1;
            }
            spaces = static_cast<uint64_t>(std::stoull(argv[++i], nullptr, 0));

        } else if (arg == "--max-acc-per-mm") {
            if (i + 1 >= argc) {
                cerr << "--max-acc-per-mm requires an argument\n";
                return 1;
            }
            max_acc_per_mm = static_cast<uint32_t>(std::stoul(argv[++i]));

        } else {
            cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    bool dna_sequence = true;
    uint64_t toggle_mask = 0xe37e28c4271b5a2dULL;
    int revcom_version = 1;
    uint64_t spaced_seed_mask = 0;

    if (spaces > 0) {
        if (l / 4 < (int)spaces) {
            cerr << "Number of spaces (" << spaces << ") exceeds max for minimizer len ("
                 << l << "); max: " << (l / 4) << "\n";
            return 1;
        }

        int num_ones = l - 2 * spaces;

        std::string template_str;
        template_str.reserve(l);

        for (int i = 0; i < num_ones; i++) {
            template_str += '1';
        }

        for (int i = 0; i < (int)spaces; i++) {
            template_str += '0';
            template_str += '1';
        }

        for (int i = 0; i < (int)template_str.length() && i < l; i++) {
            if (template_str[i] == '1') {
                spaced_seed_mask |= (1ULL << i);
            }
        }

        cout << "[PARAM] Using Kraken2 spaced seed: l=" << l
             << ", spaces=" << spaces
             << " (mask=0x" << std::hex << spaced_seed_mask << std::dec << ")\n";
    } else {
        for (int i = 0; i < l; i++) {
            spaced_seed_mask |= (1ULL << i);
        }
        cout << "[PARAM] No spaced seed (using all positions)\n";
    }

    ScannerParams params{
        k,
        l,
        spaced_seed_mask,
        dna_sequence,
        toggle_mask,
        revcom_version
    };

    cout << "[STEP] Loading accessions from: " << accessions_file << "\n";
    auto valid_accessions = load_accessions(accessions_file);
    cout << "[INFO] Loaded " << valid_accessions.size() << " accessions\n";

    FlatMinimizerMap mm_to_acc;
    vector<string> acc_list;
    unordered_map<string, uint32_t> acc_to_idx;

    cout << "[STEP] Building minimizer index from library\n";
    cout << "[INFO] Library file: " << library << "\n";
    build_minimizer_map(library, valid_accessions, params, mm_to_acc, acc_list, acc_to_idx, threads, max_acc_per_mm);
    cout << "[INFO] Indexed " << acc_list.size() << " accessions\n";
    cout << "[INFO] Total minimizer entries: " << mm_to_acc.size() << "\n";
    cout << "[INFO] Minimizer map memory: " << mm_to_acc.memory_bytes() / (1024*1024) << " MB\n";

    {
        std::filesystem::path out(out_path);
        if (out.has_parent_path()) {
            std::filesystem::create_directories(out.parent_path());
        }
    }

    unordered_set<string> allowed_read_ids;
    if (!kraken_file.empty()) {
        cout << "[STEP] Loading read IDs from Kraken file: " << kraken_file << "\n";
        get_read_ids(allowed_read_ids, kraken_file);
        cout << "[INFO] Loaded " << allowed_read_ids.size() << " read IDs\n";
    }

    cout << "[STEP] Processing reads (streaming)\n";
    cout << "[INFO] Reads file: " << reads << "\n";
    if (!paired_reads.empty()) {
        cout << "[INFO] Paired reads file: " << paired_reads << "\n";
    }

    StreamingCOOWriter writer;
    writer.open(out_path, out_path + ".row_ids");

    process_reads_streaming(reads, paired_reads, params, mm_to_acc,
                            allowed_read_ids, threads, writer);

    cout << "[STEP] Assembling binary sparse matrix: " << out_path << "\n";
    writer.assemble(out_path, out_path, static_cast<int64_t>(acc_list.size()));

    write_col_ids(acc_list, out_path);

    cout << "[STEP] Done. Wrote: " << out_path
         << ", " << out_path << ".row_ids"
         << ", " << out_path << ".col_ids\n";
    return 0;
}
