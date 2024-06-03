/**
 *
 */
#include <iostream>
#include <fstream>
#include <string>
#include <vector>


// TODO: support multiple chromosome
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

        // append '$' as end-of-sequence mark
        Sequence.append("$");
    }

public:
    std::string Name;
    std::string Sequence;
};

class CompareSuffix {
private:
    const std::string &text;

public:
    CompareSuffix(const std::string &text) : text(text) {}

    bool operator()(int i, int j) const {
        return text.substr(i) < text.substr(j);
    }
};

std::vector<int> RK; // rank array
std::vector<int> OldRK; // rank array
int w;

bool cmp_RK(int a, int b) {
    if (RK[a] != RK[b]) return RK[a] < RK[b];
    return RK[a + w] < RK[b + w];
}

class FSuffixArray {

public:
//    FSuffixArray(const std::string &text) : text(text) { }


    void BuildSuffixArray() {
        int n = Reference.Sequence.length();
        SA.resize(n);
        RK.resize(n);
        OldRK.resize(n);

        for (int i = 1; i <= n; ++i) SA[i] = i, RK[i] = Reference.Sequence[i];

//        printf("Building Suffix Array\n");
        for (w = 1; w < n; w <<= 1) {
            // sort by rank
            std::sort(SA.begin(), SA.end(), cmp_RK);
            for (int i = 1; i <= n; ++i) OldRK[i] = RK[i];  // save the old rank

            // update rank array based on the sorted suffix array
            for (int i = 1, p = 0; i <= n; ++i) {
                if (OldRK[SA[i]] == OldRK[SA[i - 1]] && OldRK[SA[i] + w] == OldRK[SA[i - 1] + w])
                    RK[SA[i]] = p;  // if the rank of SA[i] and SA[i-1] are equal, then they are in the same group
                else RK[SA[i]] = ++p;  // otherwise, they are in different groups
            }
        }
//        printf("Suffix Array:\n");
        for (int i = 1; i <= 5; ++i){
            printf("%d ", SA[i]);
            // print the first 10 characters of the SA[i]-th suffix
            printf("%s\n", Reference.Sequence.substr(SA[i], 10).c_str());
        }
    }

    void Save(const char *InFilename) {
        std::ofstream outFile(InFilename);
        if (!outFile.is_open()) {
            std::cerr << "Failed to open file for writing: " << InFilename << std::endl;
            return;
        }

//        for (size_t i = 0; i < SA.size(); i++) {
//            outFile << SA[i] << std::endl;
//        }

        for (size_t i = 0; i < 200; i++) {
            outFile << SA[i] << std::endl;
        }

        outFile.close();
    }

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
    std::vector<uint32_t> SA; // suffix array

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
    std::cerr << "  " << InProgramName << " Reference_Fasta_File SuffixArray_File" << std::endl;
}


void test_1(const char *InFilename) {
    FReference ref(InFilename);

    std::cout << "Reference sequence length: " << ref.Sequence.length() << std::endl;
    // print first 100bp
    std::cout << ref.Sequence.substr(0, 100) << std::endl;
}

void test_2() {
    FReference ref2;
    ref2.LoadFromString("AACCGTA");

    std::cout << "Reference2 sequence length: " << ref2.Sequence.length() << std::endl;

    // print first 100bp
    std::cout << ref2.Sequence.substr(0, 100) << std::endl;
}

void test_3(const char *InRefFilename, const char *InSAFilename) {
    FSuffixArray SA;

    printf("Loading reference from %s\n", InRefFilename);

    SA.Reference.LoadFromFasta(InRefFilename);

    printf("Building suffix array\n");

    SA.BuildSuffixArray();

    printf("Saving suffix array to %s\n", InSAFilename);

    SA.Save(InSAFilename);
}

/**
 * 
 *
 */
int main(int argc, char *argv[]) {
    // sa InReferenceFastaFile OutSuffixArrayFile
    if (argc < 3) {
        PrintUsage(GetFilename(argv[0]));
        return 1;
    }

    test_1(argv[1]);

    test_2();

    test_3(argv[1], argv[2]);

    return 0;
}
