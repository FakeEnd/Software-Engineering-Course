#include <iostream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <iostream>
#include <fstream>
#include <cstring>


using namespace std;

// insert code here...
// more code from previous project
class FReference {
public:
    FReference() {};

    FReference(const std::string &InFilename) {
        LoadFromFasta(InFilename);
    }

    void LoadFromString(const std::string &InSequence) {
        Sequence = InSequence;
        if (Sequence.back() != '$') {
            Sequence.push_back('$');
        }
    }

    void LoadFromFasta(const std::string &InFilename) {
        std::ifstream fs(InFilename);
        if (!fs) {
            std::cerr << "Can't open file: " << InFilename << std::endl;
            return;
        }

        std::string buf;

        Sequence.clear();

        while (getline(fs, buf)) {
            if (buf.length() == 0) {
                // skip empty line
                continue;
            }

            if (buf[0] == '>') {
                // header line
                // TODO: save chromosome name
                continue;
            }

            Sequence.append(buf);
        }

        // append '$' as End of Sequence mark
        Sequence.append("$");
    }

public:
    std::string Name;
    std::string Sequence;
};


struct Suffix {
    int index;
    std::pair<int, int> rank;
};

// Comparator function for sorting suffixes based on ranks
bool compareSuffix(const Suffix &a, const Suffix &b) {
    return a.rank < b.rank;
};


class FSuffixArray {
public:

    void BuildSuffixArray() {
        SA.clear();
        int n = Reference.Sequence.length();
        std::vector<int> rank(n);
        std::vector<Suffix> suffixes(n);
        SA.resize(n);

        // Initialize ranks and suffix indices
        for (int i = 0; i < n; ++i) {
            rank[i] = Reference.Sequence[i];
            suffixes[i].index = i;
            suffixes[i].rank = std::pair<int, int>(rank[i], i + 1 < n ? Reference.Sequence[i + 1] : -1);
        }

        // Sort based on the first two characters
        std::sort(suffixes.begin(), suffixes.end(), compareSuffix);

        for (int k = 4; k < 2 * n; k *= 2) {
            int rankValue = 0;
            std::pair<int, int> prevRank = suffixes[0].rank;
            suffixes[0].rank.first = rankValue;
            SA[suffixes[0].index] = 0;

            // Assign new rank
            for (int i = 1; i < n; ++i) {
                if (suffixes[i].rank == prevRank) {
                    suffixes[i].rank.first = rankValue;
                } else {
                    prevRank = suffixes[i].rank;
                    suffixes[i].rank.first = ++rankValue;
                }
                SA[suffixes[i].index] = i;
            }

            // Assign next rank using sorted suffix array
            for (int i = 0; i < n; ++i) {
                int nextIndex = suffixes[i].index + k / 2;
                suffixes[i].rank.second = nextIndex < n ? suffixes[SA[nextIndex]].rank.first : -1;
            }

            // Sort based on first k characters
            std::sort(suffixes.begin(), suffixes.end(), compareSuffix);
        }

        // Extract indices to form the final suffix array
        for (int i = 0; i < n; ++i) {
            SA[i] = suffixes[i].index;
        }

    }


    /**
     * Save a suffix array to file
     * @param InFilename Output filename.
     *
     * The format of .sa file is described in the homepage/README.md file.
     * Each line contains a single number that corresponds to an item in the SA.
     *
     */
    void Save(const char *InFilename) {
        std::ofstream outFile(InFilename);
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file for writing: " << InFilename << std::endl;
            return;
        }

        for (size_t i = 0; i < SA.size(); i++) {
            outFile << SA[i] << std::endl;
        }

        outFile.close();
    }

    /**
     * Load a suffix array from file
     * @param InFilename Input filename
     *
     * TIP:
     * If the symbol '$' is used as an end-of-sequence mark,
     * the first line of SA file is the index of '$' within the sequence.
     * In that case, the index would be one less than the length of the array.
     */
    void Load(const char *InFilename) {
        std::ifstream inFile(InFilename);
        if (!inFile.is_open()) {
            std::cerr << "Failed to open file for reading: " << InFilename << std::endl;
            return;
        }

        SA.clear();
        int index;
        while (inFile >> index) {
            SA.push_back(index);
        }

        inFile.close();
    }

public:
    FReference Reference;
    std::vector<uint32_t> SA;

};

/*
 * Return filename from full path of file.
 *
 * InFilename   [In] Full path of file
 *
 * Return
 *   base filename
 *
 */
static std::string GetFilename(const std::string &InFilename) {
    const size_t pos = InFilename.find_last_of("/\\");
    if (pos == std::string::npos) {
        return InFilename;
    }

    return InFilename.substr(pos + 1);
}

//Load queries
std::vector<string> LoadQueryFromFasta(const std::string &InFilename) {
    std::ifstream fs(InFilename);

    std::string buf;
    std::vector<string> queryVector;

    while (getline(fs, buf)) {
        if (buf.length() == 0) {
            // skip empty line
            continue;
        }

        if (buf[0] == '>') {
            // header line
            continue;
        }
//        printf("buf: %s\n", buf.c_str());
        queryVector.push_back(buf);
    }

    return queryVector;
}

struct FAlignResult {
    std::string ChrName;
    int Position;

    FAlignResult() : Position(0) {};

    FAlignResult(const std::string &InChrName, int InPosition)
            : ChrName(InChrName), Position(InPosition) {};
};

class FBwt {
    // insert code here
public:
    std::string bwt;
    std::map<char, std::vector<int> > rank;
    std::map<char, int> counts;
    std::vector<uint32_t> SA;

public:
    void build_bwt(const std::string &reference_seq, std::vector<uint32_t> &sa) {
        SA = sa;
        int n = reference_seq.size();

        for (int i = 0; i < n; ++i) {
            bwt += reference_seq[(sa[i] + n - 1) % n];
        }
    }

    std::vector<int> rank_table(const std::string &bwt, char c) {
        std::vector<int> rank(bwt.size() + 1, 0);
        for (int i = 1; i <= bwt.size(); ++i) {
            rank[i] = rank[i - 1] + (bwt[i - 1] == c ? 1 : 0);
        }
        return rank;
    }

    void build_rank_table() {
        for (auto &p: counts) {
            rank[p.first] = rank_table(bwt, p.first);
        }
    }

    void count_table() {
        for (char c: bwt) {
            counts[c]++;
        }
        int cum = 0;
        for (auto &p: counts) {
            int temp = p.second;
            p.second = cum;
            cum += temp;
        }
    }

    void fm_align(const std::string &InName, const std::string &InQuery, std::vector<FAlignResult> &OutResult) {
        int l = 0, r = bwt.size() - 1;
        for (int i = InQuery.size() - 1; i >= 0 && l <= r; --i) {
            char c = InQuery[i];
            l = counts.at(c) + rank.at(c)[l];
            r = counts.at(c) + rank.at(c)[r + 1] - 1;
//            printf("l: %d, r: %d\n", l, r);
        }

        for (; l <= r; ++l) {
            OutResult.push_back(FAlignResult(InName, SA[l]));
        }
    }
};


class FBwt_save_memory {
    // insert code here
public:
    std::string bwt;
    std::map<char, std::vector<int> > rank;
    std::map<char, int> counts;
    std::vector<uint32_t> SA;
    int sample_rate;

public:
    FBwt_save_memory(int sample_rate = 16) {
        // default divide number is 16
        this->sample_rate = sample_rate;
    }

    void build_bwt(const std::string &reference_seq, std::vector<uint32_t> &sa) {
        SA = sa;
        int n = reference_seq.size();

        for (int i = 0; i < n; ++i) {
            bwt += reference_seq[(sa[i] + n - 1) % n];
        }
    }

    std::vector<int> rank_table(const std::string &bwt, char c) {
        std::vector<int> rank((bwt.size() / sample_rate) + 1, 0);
        for (int i = 1, count = 0; i <= bwt.size(); ++i) {
//            rank[i] = rank[i - 1] + (bwt[i - 1] == c ? 1 : 0);
            if (bwt[i - 1] == c)count++;
            if (i % sample_rate == 0) {
                rank[i / sample_rate] = count;
            }
        }
        return rank;
    }

    void build_rank_table() {
        for (auto &p: counts) {
            rank[p.first] = rank_table(bwt, p.first);
        }

    }

    void count_table() {
        for (char c: bwt) {
            counts[c]++;
        }
        int cum = 0;
        for (auto &p: counts) {
            int temp = p.second;
            p.second = cum;
            cum += temp;
        }
    }

    void fm_align(const std::string &InName, const std::string &InQuery, std::vector<FAlignResult> &OutResult) {
        int l = 0, r = bwt.size() - 1;

        for (int i = InQuery.size() - 1; i >= 0 && l <= r; --i) {
            char c = InQuery[i];
            int newL = counts.at(c);
            int newR = counts.at(c);
//            printf("newL: %d, newR: %d\n", newL, newR);
            if (l % sample_rate != 0) {
                newL += rank.at(c)[l / sample_rate];
//                printf("newL: %d\n", newL);
                for (int j = l - l % sample_rate; j < l; j++) {
//                    printf("j: %d\n", j);
                    if (bwt[j] == c) {
                        newL = newL + 1;
                    }
                }

            } else {
                newL += rank.at(c)[l / sample_rate];
            }
//            printf("newL: %d, newR: %d\n", newL, newR);
            if ((r + 1) % sample_rate != 0) {
                newR += rank.at(c)[(r + 1) / sample_rate];
//                printf("newR: %d\n", newR);
                for (int j = r - (r + 1) % sample_rate + 1; j < (r + 1); j++) {
                    if (bwt[j] == c) {
                        newR = newR + 1;
                    }
                }
            } else {
                newR += rank.at(c)[(r + 1) / sample_rate];
            }
//            printf("newL: %d, newR: %d\n", newL, newR);
            l = newL;
            r = newR - 1;
//            printf("l: %d, r: %d\n", l, r);
        }
//        printf("final l: %d, r: %d\n", l, r);
        for (; l <= r; ++l) {
            OutResult.push_back(FAlignResult(InName, SA[l]));
        }
    }
};

class FBwt_save_memory_use_SA {
    // insert code here
public:
    std::string bwt;
    std::map<char, std::vector<int> > rank;
    std::map<char, int> counts;
    std::vector<uint32_t> SA;
    int sample_rate;

public:
    FBwt_save_memory_use_SA(int sample_rate = 16) {
        // default divide number is 16
        this->sample_rate = sample_rate;
    }

    void build_bwt(const std::string &reference_seq, std::vector<uint32_t> &sa) {
//        SA = sa;
        int n = reference_seq.size();

        for (int i = 0; i < n; ++i) {
            bwt += reference_seq[(sa[i] + n - 1) % n];
        }

        // save SA use sample_rate to save memory
        for (int i = 0; i < n; i += sample_rate) {
            SA.push_back(sa[i]);
        }
    }

    std::vector<int> rank_table(const std::string &bwt, char c) {
        std::vector<int> rank((bwt.size() / sample_rate) + 1, 0);
        for (int i = 1, count = 0; i <= bwt.size(); ++i) {
//            rank[i] = rank[i - 1] + (bwt[i - 1] == c ? 1 : 0);
            if (bwt[i - 1] == c)count++;
            if (i % sample_rate == 0) {
                rank[i / sample_rate] = count;
            }
        }
        return rank;
    }

    void build_rank_table() {
        for (auto &p: counts) {
            rank[p.first] = rank_table(bwt, p.first);
        }

    }

    void count_table() {
        for (char c: bwt) {
            counts[c]++;
        }
        int cum = 0;
        for (auto &p: counts) {
            int temp = p.second;
            p.second = cum;
            cum += temp;
        }
    }

    void fm_align(const std::string &InName, const std::string &InQuery, std::vector<FAlignResult> &OutResult) {
        int l = 0, r = bwt.size() - 1;
        for (int i = InQuery.size() - 1; i >= 0 && l <= r; --i) {
            char c = InQuery[i];
            int newL = counts.at(c);
            int newR = counts.at(c);
            if (l % sample_rate != 0) {
                newL += rank.at(c)[l / sample_rate];
                for (int j = l - l % sample_rate; j < l; j++) {
                    if (bwt[j] == c) {
                        newL = newL + 1;
                    }
                }

            } else {
                newL += rank.at(c)[l / sample_rate];
            }

            if ((r + 1) % sample_rate != 0) {
                newR += rank.at(c)[(r + 1) / sample_rate];
                for (int j = r - (r + 1) % sample_rate + 1; j < (r + 1); j++) {
                    if (bwt[j] == c) {
                        newR = newR + 1;
                    }
                }
            } else {
                newR += rank.at(c)[(r + 1) / sample_rate];
            }
            l = newL;
            r = newR - 1;
        }

//        printf("l: %d, r: %d\n", l, r);

        for (; l <= r; ++l) {
            int count = 0;
            int temp_l = l;
            while (temp_l % sample_rate != 0) {
                char c = bwt[temp_l];
                int temp_temp_l = counts.at(c) + rank.at(c)[temp_l / sample_rate];
                for (int j = temp_l - temp_l % sample_rate; j < temp_l; j++) {
                    if (bwt[j] == c) {
                        temp_temp_l = temp_temp_l + 1;
                    }
                }
                temp_l = temp_temp_l;
                count++;
            }
            OutResult.push_back(FAlignResult(InName, SA[temp_l / sample_rate] + count));
//            OutResult.push_back(FAlignResult(InName, SA[l]));
        }
    }

//    >checkpointed tallies matrix
//    0 0 0 1 0 0
//    1 0 1 1 0 0
//    1 2 1 1 0 0
//    1 3 2 1 0 0
//    >subsampled suffix array
//    6 0 1 5
//    >Burrows-Wheeler transform text
//            GC$AAAC
//    >SampleRate
//    2
    void save_bwt_fm(const char *InFilename) {
        std::ofstream outFile(InFilename);
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file for writing: " << InFilename << std::endl;
            return;
        }

        outFile << ">checkpointed tallies matrix" << std::endl;
        for (auto &p: rank) {
            outFile << p.first << " ";
            // print p.second from index 1 to 100
//            for (int i = 1; i < p.second.size(); i++) {
            for (int i = 1; i < 100 and i < p.second.size(); i++) {
                outFile << p.second[i] << " ";
            }
            outFile << std::endl;
        }

        outFile << ">subsampled suffix array" << std::endl;
//        for (auto &p: SA) {
//            outFile << p << " ";
//        }
        for (int i = 0; i < 100 and i < SA.size(); i++) {
            outFile << SA[i] << " ";
        }

        outFile << std::endl;

        outFile << ">Burrows-Wheeler transform text" << std::endl;
//        outFile << bwt.substr(0, 100) << std::endl;
        outFile << bwt << std::endl;

        outFile << ">SampleRate" << std::endl;
        outFile << sample_rate << std::endl;

        outFile.close();
    }
};


// salign <SAFileName> <RefFilename> <ReadFilename>
// Users/junrujin/SEClass/Example_Files/sample_1.aln
// Users/junrujin/SEClass/Example_Files/sample_ref.sa
// Users/junrujin/SEClass/Example_Files/sample_ref.fa
// Users/junrujin/SEClass/Example_Files/sample_reads.fa
// Users/junrujin/SEClass/Example_Files/sample.fm
int main(int argc, char *argv[]) {
    // insert code here...
    FSuffixArray sa;
    sa.Reference.LoadFromFasta(argv[3]);
    sa.Load(argv[2]);
//    sa.BuildSuffixArray();

    auto start = std::chrono::high_resolution_clock::now();


//    FBwt_save_memory bwt(16);
    FBwt_save_memory_use_SA bwt(2);
//    FBwt bwt;
    bwt.build_bwt(sa.Reference.Sequence, sa.SA);
    bwt.count_table();
    bwt.build_rank_table();
    bwt.save_bwt_fm(argv[5]);
    // print counts
//    for (auto &p: bwt.counts) {
//        printf("%c: %d\n", p.first, p.second);
//    }

    //print bwt
//    std::cout << "bwt: " << bwt.bwt << std::endl;

    //print rank table
//    for (auto &p: bwt.rank) {
//        // print p.first
//        printf("%c: ", p.first);
//        for (auto &q: p.second) {
//            printf("%d ", q);
//        }
//        printf("\n");
//    }

//    auto end = std::chrono::high_resolution_clock::now();
//    std::chrono::duration<double> elapsed = end - start;
//
//    std::cout << "bwt construct time: " << elapsed.count() << "s" << std::endl;
//
//    std::vector<string> queryVector = LoadQueryFromFasta(argv[4]);
//    // print queries
//    for (size_t i = 0; i < 5; i++) {
//        std::cout << "Query" << i << ": " << queryVector[i] << std::endl;
//    }
//
//    // compute time
//    start = std::chrono::high_resolution_clock::now();
//
//    //align
//    std::vector<FAlignResult> alignResults;
//    for (size_t i = 0; i < queryVector.size(); i++) {
////    for (size_t i = 0; i < 12; i++) {
////        if (i % 10 == 0) {
////            std::cout << "Query" << i << std::endl;
////        }
////        printf("index: %d\n", i);
//        bwt.fm_align("query" + std::to_string(i + 1), queryVector[i], alignResults);
//    }
//
//    end = std::chrono::high_resolution_clock::now();
//    elapsed = end - start;
//    std::cout << "query time: " << elapsed.count() << "s" << std::endl;
//
//    //print results
//    for (size_t i = 0; i < 5; i++) {
//        std::cout << alignResults[i].ChrName << "\t" << (alignResults[i].Position + 1) << std::endl;
//    }
//
//    //save results
//    std::ofstream outFile(argv[1]);
//    if (!outFile.is_open()) {
//        std::cerr << "Failed to open file for writing: " << argv[1] << std::endl;
//        return 1;
//    }
//
//    for (size_t i = 0; i < 500; i++) {
//        outFile << alignResults[i].ChrName << "\t" << "sample" << "\t" << (alignResults[i].Position + 1) << std::endl;
//    }

    return 0;
}

