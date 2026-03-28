#pragma once

#include "Types.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace tyre {

class DatasetImagePreprocessor {
public:
    struct ProcessedImageResult {
        std::string inputPath;
        std::string outputPath;
        bool maskFound = false;
        std::vector<NamedTiming> stepTimings;
    };

    ProcessedImageResult processFile(const std::string& inputPath, const std::string& outputDir) const;
    std::vector<ProcessedImageResult> processDirectory(const std::string& inputDir, const std::string& outputDir) const;

private:
    static bool hasSupportedImageExtension(const std::string& extension);
    static cv::Mat buildForegroundMask(const cv::Mat& gray, std::vector<NamedTiming>& timings, bool& maskFound);
    static cv::Mat resizeToTargetMegapixels(const cv::Mat& image, double targetMegaPixels);
};

}  // namespace tyre
