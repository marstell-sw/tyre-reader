#include "DatasetBenchmark.h"
#include "TyreAnalyzer.h"
#include "Types.h"

#include <filesystem>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tyre {

std::string indent(int level) {
    return std::string(static_cast<std::size_t>(level) * 2U, ' ');
}

std::string boolToJson(bool value) {
    return value ? "true" : "false";
}

std::string quote(const std::string& value) {
    return "\"" + jsonEscape(value) + "\"";
}

std::string fieldResultToJson(const FieldResult& field, bool pretty, int level) {
    const std::string i0 = pretty ? indent(level) : "";
    const std::string i1 = pretty ? indent(level + 1) : "";
    const std::string nl = pretty ? "\n" : "";
    const std::string sep = pretty ? " " : "";

    std::string json = i0 + "{" + nl;
    json += i1 + "\"raw\":" + sep + quote(field.rawText) + "," + nl;
    json += i1 + "\"normalized\":" + sep + quote(field.normalizedText) + "," + nl;
    json += i1 + "\"found\":" + sep + boolToJson(field.found) + "," + nl;
    json += i1 + "\"confidence\":" + sep + formatDouble(field.confidence) + "," + nl;
    json += i1 + "\"uncertainty\":" + sep + formatDouble(field.uncertainty) + "," + nl;
    json += i1 + "\"roiQuality\":" + sep + formatDouble(field.roiQuality) + "," + nl;
    json += i1 + "\"cropPath\":" + sep + quote(field.cropPath) + "," + nl;
    json += i1 + "\"boundingBox\":" + sep
         + quote(std::to_string(field.boundingBox.x) + "," +
                 std::to_string(field.boundingBox.y) + "," +
                 std::to_string(field.boundingBox.width) + "," +
                 std::to_string(field.boundingBox.height)) + nl;
    json += i0 + "}";
    return json;
}

std::string timingToJson(const TimingInfo& timings, bool pretty, int level) {
    const std::string i0 = pretty ? indent(level) : "";
    const std::string i1 = pretty ? indent(level + 1) : "";
    const std::string nl = pretty ? "\n" : "";
    const std::string sep = pretty ? " " : "";

    std::string json = i0 + "{" + nl;
    json += i1 + "\"preprocessMs\":" + sep + formatDouble(timings.preprocessMs) + "," + nl;
    json += i1 + "\"roiProposalMs\":" + sep + formatDouble(timings.roiProposalMs) + "," + nl;
    json += i1 + "\"ocrMs\":" + sep + formatDouble(timings.ocrMs) + "," + nl;
    json += i1 + "\"parsingMs\":" + sep + formatDouble(timings.parsingMs) + "," + nl;
    json += i1 + "\"cropSaveMs\":" + sep + formatDouble(timings.cropSaveMs) + "," + nl;
    json += i1 + "\"overlaySaveMs\":" + sep + formatDouble(timings.overlaySaveMs) + "," + nl;
    json += i1 + "\"totalMs\":" + sep + formatDouble(timings.totalMs) + nl;
    json += i0 + "}";
    return json;
}

std::string stepTimingsToJson(const std::vector<NamedTiming>& timings, bool pretty, int level) {
    const std::string i0 = pretty ? indent(level) : "";
    const std::string i1 = pretty ? indent(level + 1) : "";
    const std::string nl = pretty ? "\n" : "";
    const std::string sep = pretty ? " " : "";

    std::string json = i0 + "[";
    if (!timings.empty()) {
        json += nl;
    }
    for (std::size_t i = 0; i < timings.size(); ++i) {
        json += i1 + "{";
        if (pretty) {
            json += "\n" + indent(level + 2) + "\"name\": " + quote(timings[i].name) + ",\n";
            json += indent(level + 2) + "\"ms\": " + formatDouble(timings[i].ms) + "\n";
            json += i1 + "}";
        } else {
            json += "\"name\":" + sep + quote(timings[i].name) + ",\"ms\":" + sep + formatDouble(timings[i].ms) + "}";
        }
        if (i + 1 < timings.size()) {
            json += ",";
        }
        json += nl;
    }
    json += i0 + "]";
    return json;
}

std::string notesToJson(const std::vector<std::string>& notes, bool pretty, int level) {
    const std::string i0 = pretty ? indent(level) : "";
    const std::string i1 = pretty ? indent(level + 1) : "";
    const std::string nl = pretty ? "\n" : "";

    std::string json = i0 + "[";
    if (!notes.empty()) {
        json += nl;
    }
    for (std::size_t i = 0; i < notes.size(); ++i) {
        json += i1 + quote(notes[i]);
        if (i + 1 < notes.size()) {
            json += ",";
        }
        json += nl;
    }
    json += i0 + "]";
    return json;
}

std::string yoloDetectionsToJson(const std::vector<YoloDetection>& detections, bool pretty, int level) {
    const std::string i0 = pretty ? indent(level) : "";
    const std::string i1 = pretty ? indent(level + 1) : "";
    const std::string i2 = pretty ? indent(level + 2) : "";
    const std::string nl = pretty ? "\n" : "";
    const std::string sep = pretty ? " " : "";

    std::string json = i0 + "[";
    if (!detections.empty()) {
        json += nl;
    }
    for (std::size_t i = 0; i < detections.size(); ++i) {
        const auto& detection = detections[i];
        json += i1 + "{";
        if (pretty) {
            json += "\n";
            json += i2 + "\"label\": " + quote(detection.label) + ",\n";
            json += i2 + "\"confidence\": " + formatDouble(detection.confidence) + ",\n";
            json += i2 + "\"acceptedForOcr\": " + boolToJson(detection.acceptedForOcr) + ",\n";
            json += i2 + "\"boundingBox\": " + quote(std::to_string(detection.box.x) + "," +
                                                     std::to_string(detection.box.y) + "," +
                                                     std::to_string(detection.box.width) + "," +
                                                     std::to_string(detection.box.height)) + "\n";
            json += i1 + "}";
        } else {
            json += "\"label\":" + sep + quote(detection.label) + ","
                 "\"confidence\":" + sep + formatDouble(detection.confidence) + ","
                 "\"acceptedForOcr\":" + sep + boolToJson(detection.acceptedForOcr) + ","
                 "\"boundingBox\":" + sep + quote(std::to_string(detection.box.x) + "," +
                                                  std::to_string(detection.box.y) + "," +
                                                  std::to_string(detection.box.width) + "," +
                                                  std::to_string(detection.box.height)) + "}";
        }
        if (i + 1 < detections.size()) {
            json += ",";
        }
        json += nl;
    }
    json += i0 + "]";
    return json;
}

std::string analysisResultToJson(const AnalysisResult& result, bool pretty, int level = 0) {
    const std::string i0 = pretty ? indent(level) : "";
    const std::string i1 = pretty ? indent(level + 1) : "";
    const std::string nl = pretty ? "\n" : "";
    const std::string sep = pretty ? " " : "";

    std::string json = i0 + "{" + nl;
    json += i1 + "\"inputPath\":" + sep + quote(result.inputPath) + "," + nl;
    json += i1 + "\"frameId\":" + sep + quote(result.frameId) + "," + nl;
    json += i1 + "\"wheelFound\":" + sep + boolToJson(result.wheelFound) + "," + nl;
    json += i1 + "\"wheelCenterX\":" + sep + formatDouble(result.wheelCenterX) + "," + nl;
    json += i1 + "\"wheelCenterY\":" + sep + formatDouble(result.wheelCenterY) + "," + nl;
    json += i1 + "\"wheelInnerRadius\":" + sep + formatDouble(result.wheelInnerRadius) + "," + nl;
    json += i1 + "\"wheelOuterRadius\":" + sep + formatDouble(result.wheelOuterRadius) + "," + nl;
    json += i1 + "\"tyreSize\":" + nl + fieldResultToJson(result.tyreSize, pretty, level + 1) + "," + nl;
    json += i1 + "\"dot\":" + nl + fieldResultToJson(result.dot, pretty, level + 1) + "," + nl;
    json += i1 + "\"tyreSizeFound\":" + sep + boolToJson(result.tyreSizeFound) + "," + nl;
    json += i1 + "\"dotFound\":" + sep + boolToJson(result.dotFound) + "," + nl;
    json += i1 + "\"dotKeywordFound\":" + sep + boolToJson(result.dotKeywordFound) + "," + nl;
    json += i1 + "\"dotCodeBodyFound\":" + sep + boolToJson(result.dotCodeBodyFound) + "," + nl;
    json += i1 + "\"dotWeekYearFound\":" + sep + boolToJson(result.dotWeekYearFound) + "," + nl;
    json += i1 + "\"dotFullFound\":" + sep + boolToJson(result.dotFullFound) + "," + nl;
    json += i1 + "\"dotWeekYear\":" + sep + quote(result.dotWeekYear) + "," + nl;
    json += i1 + "\"dotFullRaw\":" + sep + quote(result.dotFullRaw) + "," + nl;
    json += i1 + "\"dotFullNormalized\":" + sep + quote(result.dotFullNormalized) + "," + nl;
    json += i1 + "\"overlayPath\":" + sep + quote(result.overlayPath) + "," + nl;
    json += i1 + "\"yoloOverlayPath\":" + sep + quote(result.yoloOverlayPath) + "," + nl;
    json += i1 + "\"debugDir\":" + sep + quote(result.debugDir) + "," + nl;
    json += i1 + "\"timingReportPath\":" + sep + quote(result.timingReportPath) + "," + nl;
    json += i1 + "\"ocrReportPath\":" + sep + quote(result.ocrReportPath) + "," + nl;
    json += i1 + "\"yoloDetections\":" + nl + yoloDetectionsToJson(result.yoloDetections, pretty, level + 1) + "," + nl;
    json += i1 + "\"notes\":" + nl + notesToJson(result.notes, pretty, level + 1) + "," + nl;
    json += i1 + "\"timings\":" + nl + timingToJson(result.timings, pretty, level + 1) + "," + nl;
    json += i1 + "\"stepTimings\":" + nl + stepTimingsToJson(result.stepTimings, pretty, level + 1) + nl;
    json += i0 + "}";
    return json;
}

std::string analysisArrayToJson(const std::vector<AnalysisResult>& results, bool pretty) {
    const std::string nl = pretty ? "\n" : "";
    std::string json = "[";
    if (!results.empty()) {
        json += nl;
    }
    for (std::size_t i = 0; i < results.size(); ++i) {
        json += analysisResultToJson(results[i], pretty, pretty ? 1 : 0);
        if (i + 1 < results.size()) {
            json += ",";
        }
        json += nl;
    }
    json += "]";
    return json;
}

std::string wheelExtractionResultToJson(const WheelExtractionResult& result, bool pretty, int level = 0) {
    const std::string i0 = pretty ? indent(level) : "";
    const std::string i1 = pretty ? indent(level + 1) : "";
    const std::string nl = pretty ? "\n" : "";
    const std::string sep = pretty ? " " : "";

    std::string json = i0 + "{" + nl;
    json += i1 + "\"inputPath\":" + sep + quote(result.inputPath) + "," + nl;
    json += i1 + "\"frameId\":" + sep + quote(result.frameId) + "," + nl;
    json += i1 + "\"wheelFound\":" + sep + boolToJson(result.wheelFound) + "," + nl;
    json += i1 + "\"wheelCenterX\":" + sep + formatDouble(result.wheelCenterX) + "," + nl;
    json += i1 + "\"wheelCenterY\":" + sep + formatDouble(result.wheelCenterY) + "," + nl;
    json += i1 + "\"wheelInnerRadius\":" + sep + formatDouble(result.wheelInnerRadius) + "," + nl;
    json += i1 + "\"wheelOuterRadius\":" + sep + formatDouble(result.wheelOuterRadius) + "," + nl;
    json += i1 + "\"originalCopyPath\":" + sep + quote(result.originalCopyPath) + "," + nl;
    json += i1 + "\"wheelOverlayPath\":" + sep + quote(result.wheelOverlayPath) + "," + nl;
    json += i1 + "\"unwrappedBandPath\":" + sep + quote(result.unwrappedBandPath) + "," + nl;
    json += i1 + "\"notes\":" + nl + notesToJson(result.notes, pretty, level + 1) + "," + nl;
    json += i1 + "\"stepTimings\":" + nl + stepTimingsToJson(result.stepTimings, pretty, level + 1) + nl;
    json += i0 + "}";
    return json;
}

std::string benchmarkSummaryToJson(const BenchmarkSummary& summary, bool pretty) {
    const std::string i0 = pretty ? indent(0) : "";
    const std::string i1 = pretty ? indent(1) : "";
    const std::string nl = pretty ? "\n" : "";
    const std::string sep = pretty ? " " : "";

    std::string json = i0 + "{" + nl;
    json += i1 + "\"totalImages\":" + sep + std::to_string(summary.totalImages) + "," + nl;
    json += i1 + "\"sizeDetectedCount\":" + sep + std::to_string(summary.sizeDetectedCount) + "," + nl;
    json += i1 + "\"sizeExactMatchCount\":" + sep + std::to_string(summary.sizeExactMatchCount) + "," + nl;
    json += i1 + "\"dotDetectedCount\":" + sep + std::to_string(summary.dotDetectedCount) + "," + nl;
    json += i1 + "\"dot4MatchCount\":" + sep + std::to_string(summary.dot4MatchCount) + "," + nl;
    json += i1 + "\"dotFullMatchCount\":" + sep + std::to_string(summary.dotFullMatchCount) + "," + nl;
    json += i1 + "\"avgSizeConfidence\":" + sep + formatDouble(summary.avgSizeConfidence) + "," + nl;
    json += i1 + "\"avgDotConfidence\":" + sep + formatDouble(summary.avgDotConfidence) + "," + nl;
    json += i1 + "\"avgTotalMs\":" + sep + formatDouble(summary.avgTotalMs) + "," + nl;
    json += i1 + "\"p50TotalMs\":" + sep + formatDouble(summary.p50TotalMs) + "," + nl;
    json += i1 + "\"p90TotalMs\":" + sep + formatDouble(summary.p90TotalMs) + "," + nl;
    json += i1 + "\"maxTotalMs\":" + sep + formatDouble(summary.maxTotalMs) + "," + nl;
    json += i1 + "\"summaryCsvPath\":" + sep + quote(summary.summaryCsvPath) + "," + nl;
    json += i1 + "\"perImageCsvPath\":" + sep + quote(summary.perImageCsvPath) + "," + nl;
    json += i1 + "\"errorsCsvPath\":" + sep + quote(summary.errorsCsvPath) + nl;
    json += i0 + "}";
    return json;
}

std::string roiOcrResultToJson(const RoiOcrResult& result, bool pretty, int level = 0) {
    const std::string i0 = pretty ? indent(level) : "";
    const std::string i1 = pretty ? indent(level + 1) : "";
    const std::string nl = pretty ? "\n" : "";
    const std::string sep = pretty ? " " : "";

    std::string json = i0 + "{" + nl;
    json += i1 + "\"imagePath\":" + sep + quote(result.imagePath) + "," + nl;
    json += i1 + "\"branch\":" + sep + quote(result.branch) + "," + nl;
    json += i1 + "\"roi\":" + sep + quote(std::to_string(result.roi.x) + "," + std::to_string(result.roi.y) + "," +
                                           std::to_string(result.roi.width) + "," + std::to_string(result.roi.height)) + "," + nl;
    json += i1 + "\"rawText\":" + sep + quote(result.rawText) + "," + nl;
    json += i1 + "\"normalizedText\":" + sep + quote(result.normalizedText) + "," + nl;
    json += i1 + "\"found\":" + sep + boolToJson(result.found) + "," + nl;
    json += i1 + "\"confidence\":" + sep + formatDouble(result.confidence) + "," + nl;
    json += i1 + "\"cropPath\":" + sep + quote(result.cropPath) + "," + nl;
    json += i1 + "\"notes\":" + nl + notesToJson(result.notes, pretty, level + 1) + "," + nl;
    json += i1 + "\"stepTimings\":" + nl + stepTimingsToJson(result.stepTimings, pretty, level + 1) + nl;
    json += i0 + "}";
    return json;
}

std::string sectorUnwrapResultToJson(const SectorUnwrapResult& result, bool pretty, int level = 0) {
    const std::string i0 = pretty ? indent(level) : "";
    const std::string i1 = pretty ? indent(level + 1) : "";
    const std::string nl = pretty ? "\n" : "";
    const std::string sep = pretty ? " " : "";

    std::string json = i0 + "{" + nl;
    json += i1 + "\"imagePath\":" + sep + quote(result.imagePath) + "," + nl;
    json += i1 + "\"branch\":" + sep + quote(result.branch) + "," + nl;
    json += i1 + "\"startAngleDeg\":" + sep + formatDouble(result.startAngleDeg) + "," + nl;
    json += i1 + "\"endAngleDeg\":" + sep + formatDouble(result.endAngleDeg) + "," + nl;
    json += i1 + "\"wheelFound\":" + sep + boolToJson(result.wheelFound) + "," + nl;
    json += i1 + "\"wheelCenterX\":" + sep + formatDouble(result.wheelCenterX) + "," + nl;
    json += i1 + "\"wheelCenterY\":" + sep + formatDouble(result.wheelCenterY) + "," + nl;
    json += i1 + "\"wheelInnerRadius\":" + sep + formatDouble(result.wheelInnerRadius) + "," + nl;
    json += i1 + "\"wheelOuterRadius\":" + sep + formatDouble(result.wheelOuterRadius) + "," + nl;
    json += i1 + "\"overlayPath\":" + sep + quote(result.overlayPath) + "," + nl;
    json += i1 + "\"unwrappedPath\":" + sep + quote(result.unwrappedPath) + "," + nl;
    json += i1 + "\"notes\":" + nl + notesToJson(result.notes, pretty, level + 1) + "," + nl;
    json += i1 + "\"stepTimings\":" + nl + stepTimingsToJson(result.stepTimings, pretty, level + 1) + nl;
    json += i0 + "}";
    return json;
}

void writeWheelGeometryReport(const std::string& path, const std::vector<WheelExtractionResult>& results) {
    std::ofstream out(path);
    for (const auto& result : results) {
        out << "image: " << result.inputPath << "\n";
        out << "frameId: " << result.frameId << "\n";
        out << "wheelFound: " << (result.wheelFound ? "true" : "false") << "\n";
        out << "originalCopy: " << result.originalCopyPath << "\n";
        out << "wheelOverlay: " << result.wheelOverlayPath << "\n";
        out << "unwrappedBand: " << result.unwrappedBandPath << "\n";
        if (!result.notes.empty()) {
            out << "notes:\n";
            for (const auto& note : result.notes) {
                out << "  - " << note << "\n";
            }
        }
        out << "timings:\n";
        for (const auto& timing : result.stepTimings) {
            out << "  " << timing.name << ": " << formatDouble(timing.ms) << " ms\n";
        }
        out << "\n";
    }
}

void printUsage() {
    std::cerr << "Usage:\n"
              << "  tyre_reader_v3 --image <file> --output <folder> [--pretty]\n"
              << "  tyre_reader_v3 --image <file> --output <folder> [--skip-ocr] [--pretty]\n"
              << "  tyre_reader_v3 --ocr-roi <file> --roi <x,y,w,h> --branch <size|dot> --output <folder> [--pretty]\n"
              << "  tyre_reader_v3 --unwrap-sector <file> --angles <start,end> --branch <size|dot> --output <folder> [--pretty]\n"
              << "  tyre_reader_v3 --wheel-image <file> --output <folder> [--pretty]\n"
              << "  tyre_reader_v3 --dir <folder> --output <folder> [--pretty]\n"
              << "  tyre_reader_v3 --dataset <dataset_root> --output <folder> [--pretty] [--debug-steps]\n"
              << "  tyre_reader_v3 --wheel-debug-dir <folder> --output <folder>\n";
}
}  // namespace tyre

int main(int argc, char** argv) {
    try {
        std::string imagePath;
        std::string ocrRoiImagePath;
        std::string unwrapSectorImagePath;
        std::string ocrRoiBranch = "size";
        std::string ocrRoiSpec;
        std::string unwrapAnglesSpec;
        std::string wheelOverrideSpec;
        std::string wheelImagePath;
        std::string dirPath;
        std::string datasetPath;
        std::string wheelDebugDirPath;
        std::string outputDir = "output";
        bool pretty = false;
        bool debugSteps = false;
        bool skipOcr = false;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--image" && i + 1 < argc) {
                imagePath = argv[++i];
            } else if (arg == "--ocr-roi" && i + 1 < argc) {
                ocrRoiImagePath = argv[++i];
            } else if (arg == "--unwrap-sector" && i + 1 < argc) {
                unwrapSectorImagePath = argv[++i];
            } else if (arg == "--roi" && i + 1 < argc) {
                ocrRoiSpec = argv[++i];
            } else if (arg == "--angles" && i + 1 < argc) {
                unwrapAnglesSpec = argv[++i];
            } else if (arg == "--wheel-override" && i + 1 < argc) {
                wheelOverrideSpec = argv[++i];
            } else if (arg == "--branch" && i + 1 < argc) {
                ocrRoiBranch = argv[++i];
            } else if (arg == "--wheel-image" && i + 1 < argc) {
                wheelImagePath = argv[++i];
            } else if (arg == "--dir" && i + 1 < argc) {
                dirPath = argv[++i];
            } else if (arg == "--dataset" && i + 1 < argc) {
                datasetPath = argv[++i];
            } else if (arg == "--wheel-debug-dir" && i + 1 < argc) {
                wheelDebugDirPath = argv[++i];
            } else if (arg == "--output" && i + 1 < argc) {
                outputDir = argv[++i];
            } else if (arg == "--pretty") {
                pretty = true;
            } else if (arg == "--debug-steps") {
                debugSteps = true;
            } else if (arg == "--skip-ocr") {
                skipOcr = true;
            } else {
                tyre::printUsage();
                return 1;
            }
        }

        const int modeCount =
            (!imagePath.empty() ? 1 : 0) + (!ocrRoiImagePath.empty() ? 1 : 0) + (!unwrapSectorImagePath.empty() ? 1 : 0) + (!wheelImagePath.empty() ? 1 : 0) +
            (!dirPath.empty() ? 1 : 0) + (!datasetPath.empty() ? 1 : 0) +
            (!wheelDebugDirPath.empty() ? 1 : 0);
        if (modeCount != 1) {
            tyre::printUsage();
            return 1;
        }

        tyre::TyreAnalyzer analyzer(debugSteps, skipOcr);
        if (!imagePath.empty()) {
            tyre::AnalysisResult result = analyzer.analyzeImageFile(imagePath, outputDir);
            result.inputPath = imagePath;
            std::cout << tyre::analysisResultToJson(result, pretty) << std::endl;
            return 0;
        }

        if (!ocrRoiImagePath.empty()) {
            int x = 0;
            int y = 0;
            int w = 0;
            int h = 0;
            if (std::sscanf(ocrRoiSpec.c_str(), "%d,%d,%d,%d", &x, &y, &w, &h) != 4) {
                tyre::printUsage();
                return 1;
            }
            const tyre::RoiOcrResult result = analyzer.recognizeRoiFile(ocrRoiImagePath, cv::Rect(x, y, w, h), ocrRoiBranch, outputDir);
            std::cout << tyre::roiOcrResultToJson(result, pretty) << std::endl;
            return 0;
        }

        if (!unwrapSectorImagePath.empty()) {
            double startAngle = 0.0;
            double endAngle = 0.0;
            double cx = 0.0;
            double cy = 0.0;
            double innerR = 0.0;
            double outerR = 0.0;
            const bool useWheelOverride = !wheelOverrideSpec.empty();
            if (std::sscanf(unwrapAnglesSpec.c_str(), "%lf,%lf", &startAngle, &endAngle) != 2) {
                tyre::printUsage();
                return 1;
            }
            if (useWheelOverride &&
                std::sscanf(wheelOverrideSpec.c_str(), "%lf,%lf,%lf,%lf", &cx, &cy, &innerR, &outerR) != 4) {
                tyre::printUsage();
                return 1;
            }
            const tyre::SectorUnwrapResult result =
                analyzer.unwrapSectorFile(unwrapSectorImagePath,
                                          ocrRoiBranch,
                                          startAngle,
                                          endAngle,
                                          useWheelOverride,
                                          cv::Point2f(static_cast<float>(cx), static_cast<float>(cy)),
                                          static_cast<float>(innerR),
                                          static_cast<float>(outerR),
                                          outputDir);
            std::cout << tyre::sectorUnwrapResultToJson(result, pretty) << std::endl;
            return 0;
        }

        if (!wheelImagePath.empty()) {
            tyre::WheelExtractionResult result = analyzer.extractWheelGeometryFile(wheelImagePath, outputDir);
            result.inputPath = wheelImagePath;
            std::cout << tyre::wheelExtractionResultToJson(result, pretty) << std::endl;
            return 0;
        }

        if (!dirPath.empty()) {
            const auto results = analyzer.analyzeDirectory(dirPath, outputDir);
            std::cout << tyre::analysisArrayToJson(results, pretty) << std::endl;
            return 0;
        }

        if (!wheelDebugDirPath.empty()) {
            const auto results = analyzer.extractWheelGeometryDirectory(wheelDebugDirPath, outputDir);
            const std::string reportPath = (std::filesystem::path(outputDir) / "wheel_geometry_timings.txt").string();
            tyre::writeWheelGeometryReport(reportPath, results);
            std::cout << "{"
                      << "\"processedImages\":" << results.size() << ","
                      << "\"reportPath\":\"" << tyre::jsonEscape(reportPath) << "\""
                      << "}" << std::endl;
            return 0;
        }

        tyre::DatasetBenchmark benchmark(analyzer);
        const tyre::BenchmarkSummary summary = benchmark.run(datasetPath, outputDir);
        std::cout << tyre::benchmarkSummaryToJson(summary, pretty) << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "{"
                  << "\"error\":\"" << tyre::jsonEscape(ex.what()) << "\""
                  << "}" << std::endl;
        return 1;
    }
}
