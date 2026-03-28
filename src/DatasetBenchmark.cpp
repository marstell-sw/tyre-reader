#include "DatasetBenchmark.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

namespace fs = std::filesystem;

namespace tyre {

namespace {

fs::path resolveDatasetImagesDir(const fs::path& root) {
    const fs::path imagesDir = root / "images";
    if (fs::exists(imagesDir) && fs::is_directory(imagesDir)) {
        return imagesDir;
    }

    const fs::path trainDir = root / "train";
    if (fs::exists(trainDir) && fs::is_directory(trainDir)) {
        return trainDir;
    }

    return {};
}

}  // namespace

DatasetBenchmark::DatasetBenchmark(TyreAnalyzer& analyzer)
    : analyzer_(analyzer) {
}

BenchmarkSummary DatasetBenchmark::run(const std::string& datasetRoot, const std::string& outputDir) {
    const fs::path root(datasetRoot);
    const fs::path imagesDir = resolveDatasetImagesDir(root);
    const fs::path expectedCsv = root / "expected.csv";

    if (imagesDir.empty() || !fs::exists(expectedCsv)) {
        throw std::runtime_error("Dataset root must contain expected.csv and either images/ or train/");
    }

    fs::create_directories(outputDir);
    const auto expectedRows = readExpectedCsv(expectedCsv.string());
    std::unordered_map<std::string, ExpectedRow> expectedByFile;
    for (const auto& row : expectedRows) {
        expectedByFile[row.filename] = row;
    }

    std::vector<BenchmarkRow> rows;
    std::vector<double> totalTimes;
    double sumSizeConfidence = 0.0;
    double sumDotConfidence = 0.0;
    double sumTotalMs = 0.0;

    for (const auto& entry : fs::directory_iterator(imagesDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const std::string filename = entry.path().filename().string();
        auto it = expectedByFile.find(filename);
        if (it == expectedByFile.end()) {
            continue;
        }

        AnalysisResult actual = analyzer_.analyzeImageFile(entry.path().string(), outputDir);
        actual.inputPath = entry.path().string();

        BenchmarkRow row;
        row.expected = it->second;
        row.actual = std::move(actual);
        row.sizeExactMatch =
            normalizeForComparison(row.expected.expectedSize) ==
            normalizeForComparison(row.actual.tyreSize.normalizedText);
        row.dot4Match =
            normalizeForComparison(row.expected.expectedDot4) ==
            normalizeForComparison(row.actual.dotWeekYear);
        row.dotFullMatch =
            normalizeForComparison(row.expected.expectedDotFull) ==
            normalizeForComparison(row.actual.dotFullNormalized);
        row.suspectedReason = classifySuspectedReason(row);
        rows.push_back(std::move(row));
    }

    BenchmarkSummary summary;
    summary.totalImages = static_cast<int>(rows.size());
    for (const auto& row : rows) {
        if (row.actual.tyreSizeFound) {
            ++summary.sizeDetectedCount;
        }
        if (row.sizeExactMatch) {
            ++summary.sizeExactMatchCount;
        }
        if (row.actual.dotFound) {
            ++summary.dotDetectedCount;
        }
        if (row.dot4Match) {
            ++summary.dot4MatchCount;
        }
        if (row.dotFullMatch) {
            ++summary.dotFullMatchCount;
        }

        sumSizeConfidence += row.actual.tyreSize.confidence;
        sumDotConfidence += row.actual.dot.confidence;
        sumTotalMs += row.actual.timings.totalMs;
        totalTimes.push_back(row.actual.timings.totalMs);
    }

    if (summary.totalImages > 0) {
        const double count = static_cast<double>(summary.totalImages);
        summary.avgSizeConfidence = sumSizeConfidence / count;
        summary.avgDotConfidence = sumDotConfidence / count;
        summary.avgTotalMs = sumTotalMs / count;
        summary.p50TotalMs = computePercentile(totalTimes, 0.50);
        summary.p90TotalMs = computePercentile(totalTimes, 0.90);
        summary.maxTotalMs = *std::max_element(totalTimes.begin(), totalTimes.end());
    }

    summary.summaryCsvPath = (fs::path(outputDir) / "summary.csv").string();
    summary.perImageCsvPath = (fs::path(outputDir) / "per_image.csv").string();
    summary.errorsCsvPath = (fs::path(outputDir) / "errors.csv").string();

    writeSummaryCsv(summary.summaryCsvPath, summary);
    writePerImageCsv(summary.perImageCsvPath, rows);
    writeErrorsCsv(summary.errorsCsvPath, rows);
    return summary;
}

std::vector<DatasetBenchmark::ExpectedRow> DatasetBenchmark::readExpectedCsv(const std::string& csvPath) {
    std::ifstream input(csvPath);
    if (!input) {
        throw std::runtime_error("Unable to open expected.csv");
    }

    std::vector<ExpectedRow> rows;
    std::string line;
    if (!std::getline(input, line)) {
        return rows;
    }

    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const auto fields = parseCsvLine(line);
        if (fields.size() < 8) {
            continue;
        }
        rows.push_back(ExpectedRow{
            fields[0], fields[1], fields[2], fields[3],
            fields[4], fields[5], fields[6], fields[7]
        });
    }

    return rows;
}

std::vector<std::string> DatasetBenchmark::parseCsvLine(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool inQuotes = false;

    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '\"') {
            if (inQuotes && i + 1 < line.size() && line[i + 1] == '\"') {
                current.push_back('\"');
                ++i;
            } else {
                inQuotes = !inQuotes;
            }
        } else if (c == ',' && !inQuotes) {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    fields.push_back(current);
    return fields;
}

std::string DatasetBenchmark::csvEscape(const std::string& value) {
    if (value.find_first_of(",\"\n\r") == std::string::npos) {
        return value;
    }
    std::string out = "\"";
    for (char c : value) {
        if (c == '\"') {
            out += "\"\"";
        } else {
            out.push_back(c);
        }
    }
    out += "\"";
    return out;
}

std::string DatasetBenchmark::classifySuspectedReason(const BenchmarkRow& row) {
    if (!row.actual.tyreSizeFound && !row.actual.dotFound) {
        return "roi_not_found";
    }
    if ((row.actual.tyreSizeFound && row.actual.tyreSize.confidence < 0.35) ||
        (row.actual.dotFound && row.actual.dot.confidence < 0.35)) {
        return "weak_ocr";
    }
    if (!row.expected.expectedDot4.empty() && row.actual.dotWeekYearFound && !row.actual.dotFullFound) {
        return "partial_dot_only";
    }
    if ((!row.expected.expectedSize.empty() && row.actual.tyreSizeFound && !row.sizeExactMatch) ||
        (!row.expected.expectedDot4.empty() && row.actual.dotFound && !row.dot4Match)) {
        return "parse_failure";
    }
    if (row.actual.tyreSize.roiQuality < 0.30 || row.actual.dot.roiQuality < 0.30) {
        return "image_quality_issue";
    }
    return "unknown";
}

void DatasetBenchmark::writeSummaryCsv(const std::string& path, const BenchmarkSummary& summary) {
    std::ofstream out(path);
    out << "metric,value\n";
    out << "totalImages," << summary.totalImages << "\n";
    out << "sizeDetectedCount," << summary.sizeDetectedCount << "\n";
    out << "sizeExactMatchCount," << summary.sizeExactMatchCount << "\n";
    out << "dotDetectedCount," << summary.dotDetectedCount << "\n";
    out << "dot4MatchCount," << summary.dot4MatchCount << "\n";
    out << "dotFullMatchCount," << summary.dotFullMatchCount << "\n";
    out << "avgSizeConfidence," << formatDouble(summary.avgSizeConfidence) << "\n";
    out << "avgDotConfidence," << formatDouble(summary.avgDotConfidence) << "\n";
    out << "avgTotalMs," << formatDouble(summary.avgTotalMs) << "\n";
    out << "p50TotalMs," << formatDouble(summary.p50TotalMs) << "\n";
    out << "p90TotalMs," << formatDouble(summary.p90TotalMs) << "\n";
    out << "maxTotalMs," << formatDouble(summary.maxTotalMs) << "\n";
}

void DatasetBenchmark::writePerImageCsv(const std::string& path, const std::vector<BenchmarkRow>& rows) {
    std::ofstream out(path);
    out << "filename,expected_size,predicted_size,expected_dot4,predicted_dot4,expected_dot_full,predicted_dot_full,"
           "size_found,size_confidence,dot_found,dot_confidence,total_ms,size_exact_match,dot4_match,dot_full_match,"
           "suspected_reason,brand,model,difficulty,notes\n";

    for (const auto& row : rows) {
        out << csvEscape(row.expected.filename) << ","
            << csvEscape(row.expected.expectedSize) << ","
            << csvEscape(row.actual.tyreSize.normalizedText) << ","
            << csvEscape(row.expected.expectedDot4) << ","
            << csvEscape(row.actual.dotWeekYear) << ","
            << csvEscape(row.expected.expectedDotFull) << ","
            << csvEscape(row.actual.dotFullNormalized) << ","
            << (row.actual.tyreSizeFound ? "true" : "false") << ","
            << formatDouble(row.actual.tyreSize.confidence) << ","
            << (row.actual.dotFound ? "true" : "false") << ","
            << formatDouble(row.actual.dot.confidence) << ","
            << formatDouble(row.actual.timings.totalMs) << ","
            << (row.sizeExactMatch ? "true" : "false") << ","
            << (row.dot4Match ? "true" : "false") << ","
            << (row.dotFullMatch ? "true" : "false") << ","
            << csvEscape(row.suspectedReason) << ","
            << csvEscape(row.expected.brand) << ","
            << csvEscape(row.expected.model) << ","
            << csvEscape(row.expected.difficulty) << ","
            << csvEscape(row.expected.notes) << "\n";
    }
}

void DatasetBenchmark::writeErrorsCsv(const std::string& path, const std::vector<BenchmarkRow>& rows) {
    std::ofstream out(path);
    out << "filename,expected_size,predicted_size,expected_dot4,predicted_dot4,expected_dot_full,predicted_dot_full,"
           "size_confidence,dot_confidence,total_ms,suspected_reason\n";

    for (const auto& row : rows) {
        if (row.sizeExactMatch && row.dot4Match && (row.expected.expectedDotFull.empty() || row.dotFullMatch)) {
            continue;
        }
        out << csvEscape(row.expected.filename) << ","
            << csvEscape(row.expected.expectedSize) << ","
            << csvEscape(row.actual.tyreSize.normalizedText) << ","
            << csvEscape(row.expected.expectedDot4) << ","
            << csvEscape(row.actual.dotWeekYear) << ","
            << csvEscape(row.expected.expectedDotFull) << ","
            << csvEscape(row.actual.dotFullNormalized) << ","
            << formatDouble(row.actual.tyreSize.confidence) << ","
            << formatDouble(row.actual.dot.confidence) << ","
            << formatDouble(row.actual.timings.totalMs) << ","
            << csvEscape(row.suspectedReason) << "\n";
    }
}

}  // namespace tyre
