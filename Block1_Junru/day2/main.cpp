#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace std;

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

struct FAlignResult {
    std::string ChrName;
    size_t Position;

    FAlignResult() : Position(0) {};

    FAlignResult(const std::string &InChrName, size_t InPosition)
            : ChrName(InChrName), Position(InPosition) {};
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

public:
    // Binary search to find all occurrences of a query in the reference sequence
    void Align(const std::string &InName, const std::string &InQuery, std::vector<FAlignResult> &OutResult) {
//        OutResult.clear();

        size_t n = Reference.Sequence.length();
        size_t m = InQuery.length();

        size_t l = 0;
        size_t r = n - 1;
        size_t p = 0;

        while (l <= r) {
            p = l + (r - l) / 2;
            int res = strncmp(InQuery.c_str(), Reference.Sequence.c_str() + SA[p], m);

            if (res == 0) {
                int left_l = l;
                int left_r = p;
                while (left_l <= left_r) {
                    int left_p = left_l + (left_r - left_l) / 2;
                    int res_left = strncmp(InQuery.c_str(), Reference.Sequence.c_str() + SA[left_p], m);
                    if (res_left == 0) {
                        left_r = left_p - 1;
                    } else {
                        left_l = left_p + 1;
                    }
                }

                int right_l = p;
                int right_r = r;
                while (right_l <= right_r) {
                    int right_p = right_l + (right_r - right_l) / 2;
                    int res_right = strncmp(InQuery.c_str(), Reference.Sequence.c_str() + SA[right_p], m);
                    if (res_right == 0) {
                        right_l = right_p + 1;
                    } else {
                        right_r = right_p - 1;
                    }
                }
                printf("left_p: %d, right_p: %d\n", left_l, right_r);
                for (int i = left_l; i <= right_r; i++) {
                    OutResult.push_back(FAlignResult(InName, SA[i]));
                }
                break;
            }

            if (res < 0) {
                r = p - 1;
            } else {
                l = p + 1;
            }
        }

        // Binary search to find the first occurrence
//        while (l <= r) {
//            p = l + (r - l) / 2;
//            int res = strncmp(InQuery.c_str(), Reference.Sequence.c_str() + SA[p], m);
//
//            if (res == 0) {
//                OutResult.push_back(FAlignResult(InName, SA[p]));
//                // find all occurrences
//                size_t i = p;
//                while (i > l && strncmp(InQuery.c_str(), Reference.Sequence.c_str() + SA[i - 1], m) == 0) {
//                    OutResult.push_back(FAlignResult(InName, SA[i - 1]));
//                    i--;
//                }
//                i = p;
//                while (i < r && strncmp(InQuery.c_str(), Reference.Sequence.c_str() + SA[i + 1], m) == 0) {
//                    OutResult.push_back(FAlignResult(InName, SA[i + 1]));
//                    i++;
//                }
//                break;
//            }
//
//            if (res < 0) {
//                r = p - 1;
//            } else {
//                l = p + 1;
//            }
//        }
    }

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

void PrintUsage(const std::string &InProgramName) {
    std::cerr << "Invalid Parameters" << std::endl;
    std::cerr << "  " << InProgramName << " SuffixArray_File Reference_Fasta_File Read_File" << std::endl;
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


// salign <SAFileName> <RefFilename> <ReadFilename>
// Users/junrujin/SEClass/Example_Files/sample_1.aln
// Users/junrujin/SEClass/Example_Files/sample_ref.sa
// Users/junrujin/SEClass/Example_Files/sample_ref.fa
// Users/junrujin/SEClass/Example_Files/sample_reads.fa
int main(int argc, char *argv[]) {
    if (argc < 4) {
        PrintUsage(GetFilename(argv[0]));
        return 1;
    }


    //create a suffix array for the sequence
    FSuffixArray sa;
    sa.Reference.LoadFromFasta(argv[3]);
//    sa.Load(argv[2]);
    sa.BuildSuffixArray();

    //print sa.SA
    for (size_t i = 0; i < sa.SA.size(); i++) {
        std::cout << sa.SA[i] << sa.Reference.Sequence.substr(sa.SA[i]) << std::endl;
    }

    //load queries
    std::vector<string> queryVector;
    queryVector = LoadQueryFromFasta(argv[4]);

    // print queries
    for (size_t i = 0; i < queryVector.size(); i++) {
        std::cout << "Query" << i << ": " << queryVector[i] << std::endl;
    }

    //align
    std::vector<FAlignResult> alignResults;
    for (size_t i = 0; i < queryVector.size(); i++) {
        sa.Align("query" + std::to_string(i + 1), queryVector[i], alignResults);
    }

    //print results
    for (size_t i = 0; i < alignResults.size(); i++) {
        std::cout << alignResults[i].ChrName << "\t" << (alignResults[i].Position + 1) << std::endl;
    }

    //save results to argv[1]
    std::ofstream outFile(argv[1]);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << argv[1] << std::endl;
        return 1;
    }

    for (size_t i = 0; i < alignResults.size(); i++) {
        outFile << alignResults[i].ChrName << "\t" << "sample" << "\t" << (alignResults[i].Position + 1) << std::endl;
    }

    return 0;
}
