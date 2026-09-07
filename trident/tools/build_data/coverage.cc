#include "mmscanner.h"
#include "kraken2_headers.h"
#include "coverage_headers.h"
#include "coverage.h"

using namespace kraken2;
using std::cerr;
using std::cout;
using std::string;
using std::ifstream;
using std::getline;
using std::unordered_set;
using std::regex;
using std::regex_search;
using std::smatch;
using std::vector;
using std::setprecision;
using std::thread;
using std::mutex;
using std::condition_variable;
using std::queue;
using std::unique_lock;
using std::lock_guard;

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

size_t count_intersection(
    const unordered_set<uint64_t>& a,
    const unordered_set<uint64_t>& b
) {
    size_t count = 0;
    for (const auto& x : a) {
        if (b.count(x)) ++count;
    }
    return count;
}

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

string extract_accession(const string& header) {
    static const regex acc_re("[A-Z]{1,2}_?\\d+\\.\\d+");
    smatch m;
    if (regex_search(header, m, acc_re))
        return m[0].str();
    size_t end = header.find_first_of(" \t");
    return header.substr(0, end);
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

void process_library(
    const std::string& library_path,
    ScannerParams& params,
    const std::unordered_set<uint64_t>& read_kmers,
    const std::string& out_path,
    int threads
) {
    std::ifstream in(library_path);
    if (!in) {
        std::cerr << "Failed to open library file\n";
        std::exit(1);
    }

    std::ofstream out(out_path);
    if (!out) {
        std::cerr << "Failed to open output\n";
        std::exit(1);
    }
    out << "#k\t" << params.k << "\n";
    out << "#l\t" << params.l << "\n";
    out << "#spaced_seed_mask\t0x" << std::hex << params.spaced_seed_mask << std::dec << "\n";
    out << "#read_kmers\t" << read_kmers.size() << "\n";
    out << "#threads\t" << threads << "\n";
    out << "accession\tseq_len\tnum_kmers\tshared_minimizers\n";

    std::mutex in_mutex;
    std::mutex out_mutex;
    std::string next_header;

    std::atomic<size_t> processed{0};

    auto worker = [&]() {
        Reference rec;

        while (true) {
            {
                std::lock_guard<std::mutex> lock(in_mutex);
                if (!read_fasta_record(in, rec, next_header))
                    return;
            }
            auto kmers = get_kmer_set(rec.seq, params);
            size_t shared = count_intersection(kmers, read_kmers);

            if (shared > 0) {
                std::lock_guard<std::mutex> lock(out_mutex);
                out << extract_accession(rec.header) << "\t"
                    << rec.seq.size() << "\t"
                    << kmers.size() << "\t"
                    << shared << "\n";
            }

            size_t c = ++processed;
            if (c % 100 == 0) {
                std::cerr << "\r[PROGRESS] references processed: "
                          << c << std::flush;
            }
        }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < threads; ++i)
        pool.emplace_back(worker);

    for (auto& t : pool)
        t.join();

    std::cerr << "\n";
}

// get_read_ids is now in coverage.h

void build_read_set(
    unordered_set<uint64_t>& read_kmers,
    const string& reads_path,
    ScannerParams& params,
    const string& paired_reads = "",
    const string& kraken_file = "",
    size_t num_threads = thread::hardware_concurrency()
) {
    unordered_set<string> allowed_read_ids;
    if (!kraken_file.empty())
        get_read_ids(allowed_read_ids, kraken_file);

    // Thread-local sets and work queue
    vector<unordered_set<uint64_t>> local_sets(num_threads);
    queue<string> work_queue;
    mutex queue_mutex;
    condition_variable cv;
    bool done = false;

    // Worker threads
    auto worker = [&](size_t tid) {
        while (true) {
            string seq;
            {
                unique_lock<mutex> lock(queue_mutex);
                cv.wait(lock, [&] { return done || !work_queue.empty(); });
                if (done && work_queue.empty()) break;
                if (!work_queue.empty()) {
                    seq = std::move(work_queue.front());
                    work_queue.pop();
                }
            }
            
            if (!seq.empty()) {
                auto kmers = get_kmer_set(seq, params);
                local_sets[tid].insert(kmers.begin(), kmers.end());
            }
        }
    };

    vector<thread> workers;
    for (size_t i = 0; i < num_threads; ++i)
        workers.emplace_back(worker, i);

    // Stream and enqueue reads
    auto process_file = [&](const string& path) {
        if (path.empty()) return;

        gzFile fp = gzopen(path.c_str(), "rb");
        if (!fp) {
            cerr << "Failed to open reads file: " << path << "\n";
            exit(1);
        }

        constexpr size_t BUF_SIZE = 1 << 20;
        char buffer[BUF_SIZE];
        string line, seq, read_id;
        bool keep = true;
        size_t reads_queued = 0;

        while (gzgets(fp, buffer, BUF_SIZE)) {
            line.assign(buffer);
            if (!line.empty() && line.back() == '\n')
                line.pop_back();
            if (line.empty()) continue;

            if (line[0] == '>' || line[0] == '@') {
                // Process previous sequence
                if (!seq.empty() && keep) {
                    {
                        lock_guard<mutex> lock(queue_mutex);
                        work_queue.push(std::move(seq));
                    }
                    cv.notify_one();
                    ++reads_queued;
                }

                seq.clear();
                
                // Extract read ID and check filter
                size_t end = line.find_first_of(" \t", 1);
                read_id = line.substr(1, end == string::npos ? string::npos : end - 1);
                keep = allowed_read_ids.empty() || allowed_read_ids.count(read_id);

            } else if (line[0] == '+') {
                // Skip FASTQ quality scores
                size_t qlen = 0;
                while (qlen < seq.size() && gzgets(fp, buffer, BUF_SIZE)) {
                    qlen += std::strlen(buffer) - (buffer[strlen(buffer)-1] == '\n');
                }
            } else {
                if (keep)
                    seq += line;
            }
        }

        // Process final sequence
        if (!seq.empty() && keep) {
            {
                lock_guard<mutex> lock(queue_mutex);
                work_queue.push(std::move(seq));
            }
            cv.notify_one();
            ++reads_queued;
        }

        cout << "[READS] " << reads_queued << " reads from " << path << "\n";
        gzclose(fp);
    };

    process_file(reads_path);
    process_file(paired_reads);

    // Signal completion
    {
        lock_guard<mutex> lock(queue_mutex);
        done = true;
    }
    cv.notify_all();

    // Wait for workers
    for (auto& t : workers)
        t.join();

    // Merge thread-local sets
    for (auto& s : local_sets)
        read_kmers.insert(s.begin(), s.end());
}

int main(int argc, char** argv) {
    if (argc < 4) {
        cerr << "Usage: ./kmer_coverage <reads> <library> <output_tsv_path> "
                "[--kraken-file FILE] "
                "[--pair paired_reads] "
                "[--threads N] "
                "[--k K] "
                "[--l L] "
                "[--spaces N]\n";
        return 1;
    }
    
    string reads = argv[1];
    string library = argv[2];
    string out_path = argv[3];
    
    string kraken_file;
    string paired_reads;
    
    int threads = 1;
    ssize_t k = 35;
    ssize_t l = 31;
    uint64_t spaces = 7;
    
    for (int i = 4; i < argc; ++i) {
        string arg = argv[i];
    
        if (arg == "--kraken-file") {
            if (i + 1 >= argc) {
                cerr << "--kraken-file requires an argument\n";
                return 1;
            }
            kraken_file = argv[++i];
    
        } else if (arg == "--pair") {
            if (i + 1 >= argc) {
                cerr << "--pair requires an argument\n";
                return 1;
            }
            paired_reads = argv[++i];
    
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
            cout << "Number of spaces (" << spaces << ") exceeds max for minimizer len (" 
                 << l << "); max: " << (l / 4) << "\n";
            return 1;
        }
        
        int num_ones = l - 2 * spaces;
        
        std::string template_str;
        template_str.reserve(l);
        
        for (int i = 0; i < num_ones; i++) {
            template_str += '1';
        }
        
        for (int i = 0; i < spaces; i++) {
            template_str += '0';
            template_str += '1';
        }
        
        for (int i = 0; i < (int)template_str.length() && i < l; i++) {
            if (template_str[i] == '1') {
                spaced_seed_mask |= (1ULL << i);
            }
        }
        
        cout << "[INFO] Using Kraken2 spaced seed: l=" << l 
             << ", spaces=" << spaces 
             << " (mask=0x" << std::hex << spaced_seed_mask << std::dec << ")\n";
    } else {
        for (int i = 0; i < l; i++) {
            spaced_seed_mask |= (1ULL << i);
        }
        cout << "[PARAM] No spaced seed (using all positions)\n";
    }

    unordered_set<uint64_t> read_kmers;
    ScannerParams params{
        k, 
        l, 
        spaced_seed_mask, 
        dna_sequence,
        toggle_mask,
        revcom_version
    };
    
    cout << "[STEP] Building read minimizer set\n";
    cout << "[INFO] Reads file: " << reads << "\n";
    if (!paired_reads.empty()) {
        cout << "[INFO] Paired reads file: " << paired_reads << "\n";
    }
    
    build_read_set(read_kmers, reads, params, paired_reads, kraken_file, threads);

    cout << "[STEP] Finished building read set\n";
    cout << "[INFO] Unique read minimizers: "
         << read_kmers.size() << "\n";
    
    cout << "[STEP] Processing reference library\n";
    cout << "[INFO] Library file: " << library << "\n";

    {
        std::filesystem::path out(out_path);
        if (out.has_parent_path()) {
            std::filesystem::create_directories(out.parent_path());
        }
    }
    
    process_library(library, params, read_kmers, out_path, threads);
    cout << "[STEP] Finished processing library\n";
    cout << "[STEP] Wrote to: " << out_path << "\n";
    return 0;
}
