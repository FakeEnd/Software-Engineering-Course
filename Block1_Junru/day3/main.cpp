#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <algorithm>
#include <vector>

using namespace std;

class FBwt
{
public:
    char* bwt_arr;

public:

    void Compute_BWT(char* refernece,
              std::vector<uint32_t> suffix_arr, int n)
    {
        // Iterates over the suffix array to find
        // the last char of each cyclic rotation
        bwt_arr = (char*)malloc(n * sizeof(char));

        int i;
        for (i = 0; i < n; i++) {
            // Computes the last char which is given by
            // input_text[(suffix_arr[i] + n - 1) % n]
            int j = suffix_arr[i] - 1;
            if (j < 0)
                j = j + n;

            bwt_arr[i] = refernece[j];
        }

        bwt_arr[i] = '\0';
    }
};




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
    std::cerr << "  " << InProgramName << " bwt Reference_Fasta_File Read_File" << std::endl;
}


// bwt <SAFileName> <RefFilename>
// Users/junrujin/SEClass/Example_Files/sample_1.bwt
// Users/junrujin/SEClass/Example_Files/sample_ref.sa
// Users/junrujin/SEClass/Example_Files/sample_ref.fa
int main(int argc, char *argv[]) {
    if (argc < 3) {
        PrintUsage(GetFilename(argv[0]));
        return 1;
    }

    //create a suffix array for the sequence
    FSuffixArray sa;
    sa.Reference.LoadFromFasta(argv[3]);
    sa.Load(argv[2]);

    //create a BWT object
    FBwt bwt;
    bwt.Compute_BWT((char*)sa.Reference.Sequence.c_str(), sa.SA, sa.Reference.Sequence.length());

    //save results to argv[1]
    std::ofstream outFile(argv[1]);
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file for writing: " << argv[1] << std::endl;
        return 1;
    }

    outFile << bwt.bwt_arr;


    return 0;
}
