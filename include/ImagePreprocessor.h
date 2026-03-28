#pragma once

#include "Types.h"

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace tyre {

class ImagePreprocessor {
public:
    cv::Mat toGrayscale(const cv::Mat& input) const;
    cv::Mat resizeUpscale(const cv::Mat& input, double scale = 2.0) const;
    cv::Mat applyClahe(const cv::Mat& gray) const;
    cv::Mat bilateralDenoise(const cv::Mat& gray) const;
    cv::Mat adaptiveThresholdImage(const cv::Mat& gray) const;
    cv::Mat invertImage(const cv::Mat& input) const;
    cv::Mat morphologyClose(const cv::Mat& input, int width, int height) const;
    cv::Mat morphologyOpen(const cv::Mat& input, int width, int height) const;
    cv::Mat deskewLight(const cv::Mat& gray) const;

    double estimateSharpness(const cv::Mat& gray) const;
    double computeImageQualityScore(const cv::Mat& gray) const;

    std::vector<std::pair<std::string, cv::Mat>> buildOcrVariants(const cv::Mat& roi) const;
    std::vector<CandidateRoi> proposeTextRegions(const cv::Mat& image) const;

private:
    static std::vector<cv::Rect> mergeNearbyBoxes(const std::vector<cv::Rect>& boxes,
                                                  const cv::Size& imageSize);
};

}  // namespace tyre
