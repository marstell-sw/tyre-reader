#pragma once

#include "TyreAnalyzer.h"
#include "Types.h"

#include <string>
#include <vector>

namespace tyre {

class DatasetBenchmark {
public:
    explicit DatasetBenchmark(TyreAnalyzer& analyzer);

    BenchmarkSummary run(const std::string& datasetRoot, const std::string& outputDir);

private:
    struct ExpectedRow {
        std::string filename;
        std::string expectedSize;
        std::string expectedDot4;
        std::string expectedDotFull;
        std::string brand;
        std::string model;
        std::string difficulty;
        std::string notes;
    };

    struct BenchmarkRow {
        ExpectedRow expected;
        AnalysisResult actual;
        bool sizeExactMatch = false;
        bool dot4Match = false;
        bool dotFullMatch = false;
        std::string suspectedReason;
    };

    TyreAnalyzer& analyzer_;

    static std::vector<ExpectedRow> readExpectedCsv(const std::string& csvPath);
    static std::vector<std::string> parseCsvLine(const std::string& line);
    static std::string csvEscape(const std::string& value);
    static std::string classifySuspectedReason(const BenchmarkRow& row);
    static void writeSummaryCsv(const std::string& path, const BenchmarkSummary& summary);
    static void writePerImageCsv(const std::string& path, const std::vector<BenchmarkRow>& rows);
    static void writeErrorsCsv(const std::string& path, const std::vector<BenchmarkRow>& rows);
};

}  // namespace tyre
