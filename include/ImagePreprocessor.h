#pragma once

#include "Types.h"

#include <opencv2/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace tyre {

class ImagePreprocessor {
public:
    struct WheelGeometry {
        bool found = false;
        cv::Point2f center;
        float radius = 0.0F;
        float innerRadius = 0.0F;
        float outerRadius = 0.0F;
        cv::Rect bounds;
        cv::RotatedRect innerEllipse;
        cv::RotatedRect outerEllipse;
    };

    struct RoiDebugImages {
        cv::Mat gray;
        cv::Mat clahe;
        cv::Mat denoised;
        cv::Mat absGradX;
        cv::Mat morph;
        cv::Mat threshold;
    };

    struct WheelDebugImages {
        cv::Mat gray;
        cv::Mat blurred;
        cv::Mat darkMask;
        cv::Mat contourOverlay;
        cv::Mat circlesOverlay;
        cv::Mat annulusOverlay;
        cv::Mat polarFull;
        cv::Mat sidewallBand;
    };

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
    WheelGeometry detectWheelGeometry(const cv::Mat& image,
                                      WheelDebugImages* debugImages = nullptr,
                                      std::vector<NamedTiming>* timings = nullptr) const;
    cv::Mat unwrapSidewallBand(const cv::Mat& image,
                               const WheelGeometry& geometry,
                               WheelDebugImages* debugImages = nullptr,
                               std::vector<NamedTiming>* timings = nullptr) const;

    std::vector<std::pair<std::string, cv::Mat>> buildOcrVariants(const cv::Mat& roi) const;
    std::vector<CandidateRoi> proposeTextRegions(const cv::Mat& image,
                                                 RoiDebugImages* debugImages = nullptr,
                                                 std::vector<NamedTiming>* timings = nullptr) const;

private:
    static std::vector<cv::Rect> mergeNearbyBoxes(const std::vector<cv::Rect>& boxes,
                                                  const cv::Size& imageSize);
};

}  // namespace tyre
