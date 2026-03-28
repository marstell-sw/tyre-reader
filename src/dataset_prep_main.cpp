#include "DatasetImagePreprocessor.h"
#include "Types.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    try {
        std::string inputDir;
        std::string outputDir;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--input-dir" && i + 1 < argc) {
                inputDir = argv[++i];
            } else if (arg == "--output-dir" && i + 1 < argc) {
                outputDir = argv[++i];
            } else {
                std::cerr << "Usage: tyre_dataset_prep --input-dir <folder> --output-dir <folder>\n";
                return 1;
            }
        }

        if (inputDir.empty() || outputDir.empty()) {
            std::cerr << "Usage: tyre_dataset_prep --input-dir <folder> --output-dir <folder>\n";
            return 1;
        }

        tyre::DatasetImagePreprocessor preprocessor;
        const auto results = preprocessor.processDirectory(inputDir, outputDir);

        const fs::path reportPath = fs::path(outputDir) / "preprocess_timings.txt";
        std::ofstream report(reportPath);
        for (const auto& result : results) {
            report << "image: " << result.inputPath << "\n";
            report << "output: " << result.outputPath << "\n";
            report << "maskFound: " << (result.maskFound ? "true" : "false") << "\n";
            report << "timings:\n";
            for (const auto& timing : result.stepTimings) {
                report << "  " << timing.name << ": " << tyre::formatDouble(timing.ms) << " ms\n";
            }
            report << "\n";
        }

        std::cout << "{"
                  << "\"processedImages\":" << results.size() << ","
                  << "\"reportPath\":\"" << tyre::jsonEscape(reportPath.string()) << "\""
                  << "}" << std::endl;
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "{"
                  << "\"error\":\"" << tyre::jsonEscape(ex.what()) << "\""
                  << "}" << std::endl;
        return 1;
    }
}
