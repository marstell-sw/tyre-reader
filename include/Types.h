#pragma once

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <opencv2/core.hpp>
#include <sstream>
#include <string>
#include <vector>

namespace tyre {

struct OcrResult {
    std::string variantName;
    std::string text;
    double averageConfidence = 0.0;
};

struct CandidateRoi {
    cv::Rect box;
    double geometryScore = 0.0;
    double imageQualityScore = 0.0;
};

struct FieldResult {
    std::string rawText;
    std::string normalizedText;
    bool found = false;
    double confidence = 0.0;
    double uncertainty = 1.0;
    double roiQuality = 0.0;
    cv::Rect boundingBox;
    std::string cropPath;
};

struct TimingInfo {
    double preprocessMs = 0.0;
    double roiProposalMs = 0.0;
    double ocrMs = 0.0;
    double parsingMs = 0.0;
    double cropSaveMs = 0.0;
    double overlaySaveMs = 0.0;
    double totalMs = 0.0;
};

struct AnalysisResult {
    std::string inputPath;
    std::string frameId;

    FieldResult tyreSize;
    FieldResult dot;

    bool tyreSizeFound = false;
    bool dotFound = false;
    bool dotWeekYearFound = false;
    bool dotFullFound = false;

    std::string dotWeekYear;
    std::string dotFullRaw;
    std::string dotFullNormalized;

    std::string overlayPath;
    std::vector<std::string> notes;
    TimingInfo timings;
};

struct BenchmarkSummary {
    int totalImages = 0;
    int sizeDetectedCount = 0;
    int sizeExactMatchCount = 0;
    int dotDetectedCount = 0;
    int dot4MatchCount = 0;
    int dotFullMatchCount = 0;
    double avgSizeConfidence = 0.0;
    double avgDotConfidence = 0.0;
    double avgTotalMs = 0.0;
    double p50TotalMs = 0.0;
    double p90TotalMs = 0.0;
    double maxTotalMs = 0.0;
    std::string summaryCsvPath;
    std::string perImageCsvPath;
    std::string errorsCsvPath;
};

inline double clamp01(double value) {
    return std::max(0.0, std::min(1.0, value));
}

inline std::string jsonEscape(const std::string& value) {
    std::ostringstream oss;
    for (unsigned char c : value) {
        switch (c) {
            case '\"': oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\b': oss << "\\b"; break;
            case '\f': oss << "\\f"; break;
            case '\n': oss << "\\n"; break;
            case '\r': oss << "\\r"; break;
            case '\t': oss << "\\t"; break;
            default:
                if (c < 0x20) {
                    oss << "\\u"
                        << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(c)
                        << std::dec << std::setfill(' ');
                } else {
                    oss << static_cast<char>(c);
                }
                break;
        }
    }
    return oss.str();
}

inline std::string formatDouble(double value, int precision = 4) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << value;
    return oss.str();
}

inline std::string normalizeForComparison(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::toupper(c)));
        }
    }
    return out;
}

inline double computePercentile(std::vector<double> values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }

    std::sort(values.begin(), values.end());
    const double position = clamp01(percentile) * static_cast<double>(values.size() - 1);
    const std::size_t lowerIndex = static_cast<std::size_t>(std::floor(position));
    const std::size_t upperIndex = static_cast<std::size_t>(std::ceil(position));
    if (lowerIndex == upperIndex) {
        return values[lowerIndex];
    }

    const double weight = position - static_cast<double>(lowerIndex);
    return values[lowerIndex] * (1.0 - weight) + values[upperIndex] * weight;
}

}  // namespace tyre
