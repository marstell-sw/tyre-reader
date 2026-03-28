#include "ImagePreprocessor.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>

namespace tyre {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

}  // namespace

cv::Mat ImagePreprocessor::toGrayscale(const cv::Mat& input) const {
    if (input.empty()) {
        return {};
    }

    cv::Mat gray;
    if (input.channels() == 1) {
        gray = input.clone();
    } else {
        cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    }
    return gray;
}

cv::Mat ImagePreprocessor::resizeUpscale(const cv::Mat& input, double scale) const {
    if (input.empty()) {
        return {};
    }

    cv::Mat resized;
    cv::resize(input, resized, cv::Size(), scale, scale, scale > 1.0 ? cv::INTER_CUBIC : cv::INTER_AREA);
    return resized;
}

cv::Mat ImagePreprocessor::applyClahe(const cv::Mat& gray) const {
    cv::Mat out;
    auto clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
    clahe->apply(gray, out);
    return out;
}

cv::Mat ImagePreprocessor::bilateralDenoise(const cv::Mat& gray) const {
    cv::Mat out;
    cv::bilateralFilter(gray, out, 7, 35.0, 35.0);
    return out;
}

cv::Mat ImagePreprocessor::adaptiveThresholdImage(const cv::Mat& gray) const {
    cv::Mat out;
    cv::adaptiveThreshold(gray, out, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 31, 7);
    return out;
}

cv::Mat ImagePreprocessor::invertImage(const cv::Mat& input) const {
    cv::Mat out;
    cv::bitwise_not(input, out);
    return out;
}

cv::Mat ImagePreprocessor::morphologyClose(const cv::Mat& input, int width, int height) const {
    cv::Mat out;
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(std::max(1, width), std::max(1, height)));
    cv::morphologyEx(input, out, cv::MORPH_CLOSE, kernel);
    return out;
}

cv::Mat ImagePreprocessor::morphologyOpen(const cv::Mat& input, int width, int height) const {
    cv::Mat out;
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(std::max(1, width), std::max(1, height)));
    cv::morphologyEx(input, out, cv::MORPH_OPEN, kernel);
    return out;
}

cv::Mat ImagePreprocessor::deskewLight(const cv::Mat& gray) const {
    if (gray.empty()) {
        return {};
    }

    cv::Mat bw;
    cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    cv::bitwise_not(bw, bw);

    std::vector<cv::Point> points;
    cv::findNonZero(bw, points);
    if (points.size() < 150) {
        return gray.clone();
    }

    const cv::RotatedRect rect = cv::minAreaRect(points);
    double angle = rect.angle;
    if (angle < -45.0) {
        angle += 90.0;
    }
    if (std::abs(angle) > 12.0) {
        return gray.clone();
    }

    const cv::Point2f center(static_cast<float>(gray.cols) / 2.0F, static_cast<float>(gray.rows) / 2.0F);
    const cv::Mat transform = cv::getRotationMatrix2D(center, angle, 1.0);
    cv::Mat rotated;
    cv::warpAffine(gray, rotated, transform, gray.size(), cv::INTER_CUBIC, cv::BORDER_REPLICATE);
    return rotated;
}

double ImagePreprocessor::estimateSharpness(const cv::Mat& gray) const {
    if (gray.empty()) {
        return 0.0;
    }

    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

double ImagePreprocessor::computeImageQualityScore(const cv::Mat& gray) const {
    if (gray.empty()) {
        return 0.0;
    }

    cv::Scalar mean;
    cv::Scalar stddev;
    cv::meanStdDev(gray, mean, stddev);
    const double sharpnessScore = std::min(estimateSharpness(gray) / 2500.0, 1.0);
    const double contrastScore = std::min(stddev[0] / 64.0, 1.0);
    return clamp01(0.55 * sharpnessScore + 0.45 * contrastScore);
}

ImagePreprocessor::WheelGeometry ImagePreprocessor::detectWheelGeometry(
    const cv::Mat& image,
    WheelDebugImages* debugImages,
    std::vector<NamedTiming>* timings) const {
    WheelGeometry geometry;
    if (image.empty()) {
        return geometry;
    }

    const Clock::time_point grayStart = Clock::now();
    const cv::Mat gray = toGrayscale(image);
    if (timings != nullptr) {
        timings->push_back({"wheel_gray_ms", elapsedMs(grayStart, Clock::now())});
    }

    const double scale = image.cols > 960 ? 960.0 / static_cast<double>(image.cols) : 1.0;
    cv::Mat resizedGray;
    if (scale < 0.999) {
        cv::resize(gray, resizedGray, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        resizedGray = gray;
    }

    const Clock::time_point blurStart = Clock::now();
    cv::Mat blurred;
    cv::medianBlur(resizedGray, blurred, 5);
    if (timings != nullptr) {
        timings->push_back({"wheel_blur_ms", elapsedMs(blurStart, Clock::now())});
    }

    const Clock::time_point houghStart = Clock::now();
    std::vector<cv::Vec3f> circles;
    const int minDim = std::min(blurred.cols, blurred.rows);
    cv::HoughCircles(blurred,
                     circles,
                     cv::HOUGH_GRADIENT,
                     1.5,
                     minDim / 6.0,
                     100.0,
                     36.0,
                     static_cast<int>(minDim * 0.18),
                     static_cast<int>(minDim * 0.52));
    if (timings != nullptr) {
        timings->push_back({"wheel_hough_ms", elapsedMs(houghStart, Clock::now())});
    }

    if (debugImages != nullptr) {
        debugImages->gray = gray;
        debugImages->blurred = blurred;
    }

    if (circles.empty()) {
        return geometry;
    }

    const cv::Point2f imageCenter(static_cast<float>(blurred.cols) * 0.5F,
                                  static_cast<float>(blurred.rows) * 0.5F);
    double bestScore = -1.0;
    cv::Vec3f bestCircle;
    for (const auto& circle : circles) {
        const cv::Point2f center(circle[0], circle[1]);
        const float radius = circle[2];
        const double centerDistance = cv::norm(center - imageCenter) / std::max(1.0, static_cast<double>(minDim));
        const double normalizedRadius = radius / std::max(1.0, static_cast<double>(minDim));
        const double score = normalizedRadius * 2.0 - centerDistance;
        if (score > bestScore) {
            bestScore = score;
            bestCircle = circle;
        }
    }

    geometry.found = true;
    geometry.center = cv::Point2f(bestCircle[0] / static_cast<float>(scale),
                                  bestCircle[1] / static_cast<float>(scale));
    geometry.radius = bestCircle[2] / static_cast<float>(scale);
    geometry.innerRadius = geometry.radius * 0.68F;
    geometry.outerRadius = geometry.radius * 0.98F;
    geometry.bounds = cv::Rect(
        std::max(0, static_cast<int>(std::floor(geometry.center.x - geometry.outerRadius))),
        std::max(0, static_cast<int>(std::floor(geometry.center.y - geometry.outerRadius))),
        std::min(image.cols, static_cast<int>(std::ceil(geometry.outerRadius * 2.0F))),
        std::min(image.rows, static_cast<int>(std::ceil(geometry.outerRadius * 2.0F))));

    if (debugImages != nullptr) {
        cv::Mat circlesOverlay;
        cv::cvtColor(resizedGray, circlesOverlay, cv::COLOR_GRAY2BGR);
        for (const auto& circle : circles) {
            cv::circle(circlesOverlay,
                       cv::Point(cvRound(circle[0]), cvRound(circle[1])),
                       cvRound(circle[2]),
                       cv::Scalar(255, 200, 0),
                       2,
                       cv::LINE_AA);
        }
        cv::circle(circlesOverlay,
                   cv::Point(cvRound(bestCircle[0]), cvRound(bestCircle[1])),
                   cvRound(bestCircle[2]),
                   cv::Scalar(0, 255, 0),
                   3,
                   cv::LINE_AA);
        debugImages->circlesOverlay = circlesOverlay;

        cv::Mat annulusOverlay = image.clone();
        cv::circle(annulusOverlay, geometry.center, cvRound(geometry.outerRadius), cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        cv::circle(annulusOverlay, geometry.center, cvRound(geometry.innerRadius), cv::Scalar(0, 140, 255), 3, cv::LINE_AA);
        debugImages->annulusOverlay = annulusOverlay;
    }

    return geometry;
}

cv::Mat ImagePreprocessor::unwrapSidewallBand(const cv::Mat& image,
                                              const WheelGeometry& geometry,
                                              WheelDebugImages* debugImages,
                                              std::vector<NamedTiming>* timings) const {
    if (image.empty() || !geometry.found || geometry.outerRadius <= geometry.innerRadius) {
        return {};
    }

    const Clock::time_point polarStart = Clock::now();
    const int angleSamples = std::clamp(static_cast<int>(std::round(geometry.outerRadius * 3.2F)), 720, 1800);
    const int radialSamples = std::clamp(static_cast<int>(std::ceil(geometry.outerRadius)), 64, 220);

    cv::Mat polar;
    cv::warpPolar(image,
                  polar,
                  cv::Size(radialSamples, angleSamples),
                  geometry.center,
                  geometry.outerRadius,
                  cv::WARP_POLAR_LINEAR);
    if (timings != nullptr) {
        timings->push_back({"wheel_warp_polar_ms", elapsedMs(polarStart, Clock::now())});
    }

    const int innerIndex = std::clamp(static_cast<int>(std::floor(geometry.innerRadius)), 0, std::max(0, polar.cols - 2));
    const int outerIndex = std::clamp(static_cast<int>(std::ceil(geometry.outerRadius)), innerIndex + 1, polar.cols);
    cv::Mat sidewall = polar.colRange(innerIndex, outerIndex).clone();
    cv::transpose(sidewall, sidewall);
    cv::flip(sidewall, sidewall, 0);
    if (sidewall.rows < 96) {
        const double scale = 96.0 / std::max(1, sidewall.rows);
        cv::resize(sidewall, sidewall, cv::Size(), scale, scale, cv::INTER_CUBIC);
    }
    if (sidewall.cols > 2200) {
        const double scale = 2200.0 / static_cast<double>(sidewall.cols);
        cv::resize(sidewall, sidewall, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    if (debugImages != nullptr) {
        debugImages->polarFull = polar;
        debugImages->sidewallBand = sidewall;
    }
    return sidewall;
}

std::vector<std::pair<std::string, cv::Mat>> ImagePreprocessor::buildOcrVariants(const cv::Mat& roi) const {
    std::vector<std::pair<std::string, cv::Mat>> variants;
    if (roi.empty()) {
        return variants;
    }

    const cv::Mat gray = toGrayscale(roi);
    const cv::Mat deskewed = deskewLight(gray);
    const cv::Mat upscaled = resizeUpscale(deskewed, 2.0);
    const cv::Mat clahe = applyClahe(upscaled);
    const cv::Mat denoised = bilateralDenoise(clahe);
    const cv::Mat adaptive = adaptiveThresholdImage(denoised);
    const cv::Mat inverted = invertImage(adaptive);
    const cv::Mat closed = morphologyClose(adaptive, 5, 3);
    const cv::Mat opened = morphologyOpen(closed, 3, 3);

    variants.emplace_back("gray_upscaled", upscaled);
    variants.emplace_back("clahe", clahe);
    variants.emplace_back("adaptive", adaptive);
    variants.emplace_back("adaptive_inverted", inverted);
    variants.emplace_back("opened", opened);
    return variants;
}

std::vector<CandidateRoi> ImagePreprocessor::proposeTextRegions(const cv::Mat& image,
                                                                RoiDebugImages* debugImages,
                                                                std::vector<NamedTiming>* timings) const {
    std::vector<CandidateRoi> output;
    if (image.empty()) {
        return output;
    }

    const Clock::time_point grayStart = Clock::now();
    const cv::Mat gray = toGrayscale(image);
    if (timings != nullptr) {
        timings->push_back({"roi_gray_ms", elapsedMs(grayStart, Clock::now())});
    }

    const Clock::time_point claheStart = Clock::now();
    const cv::Mat clahe = applyClahe(gray);
    if (timings != nullptr) {
        timings->push_back({"roi_clahe_ms", elapsedMs(claheStart, Clock::now())});
    }

    const Clock::time_point denoiseStart = Clock::now();
    const cv::Mat denoised = bilateralDenoise(clahe);
    if (timings != nullptr) {
        timings->push_back({"roi_bilateral_ms", elapsedMs(denoiseStart, Clock::now())});
    }

    const Clock::time_point sobelStart = Clock::now();
    cv::Mat gradX;
    cv::Sobel(denoised, gradX, CV_32F, 1, 0, 3);
    cv::Mat absGradX;
    cv::convertScaleAbs(gradX, absGradX);
    if (timings != nullptr) {
        timings->push_back({"roi_sobel_ms", elapsedMs(sobelStart, Clock::now())});
    }

    const Clock::time_point morphStart = Clock::now();
    cv::Mat morph = morphologyClose(absGradX, std::max(15, image.cols / 25), std::max(3, image.rows / 120));
    cv::Mat thresh;
    cv::threshold(morph, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    thresh = morphologyClose(thresh, std::max(17, image.cols / 20), std::max(3, image.rows / 100));
    thresh = morphologyOpen(thresh, 3, 3);
    if (timings != nullptr) {
        timings->push_back({"roi_morph_threshold_ms", elapsedMs(morphStart, Clock::now())});
    }

    const Clock::time_point contourStart = Clock::now();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (timings != nullptr) {
        timings->push_back({"roi_contours_ms", elapsedMs(contourStart, Clock::now())});
    }

    const Clock::time_point filterStart = Clock::now();
    std::vector<cv::Rect> rawBoxes;
    const double imageArea = static_cast<double>(image.cols) * static_cast<double>(image.rows);
    for (const auto& contour : contours) {
        const cv::Rect rect = cv::boundingRect(contour);
        const double area = static_cast<double>(rect.area());
        if (area < imageArea * 0.0004 || area > imageArea * 0.28) {
            continue;
        }
        if (rect.width < 40 || rect.height < 12) {
            continue;
        }
        const double aspect = static_cast<double>(rect.width) / std::max(1, rect.height);
        if (aspect < 1.6 || aspect > 18.0) {
            continue;
        }
        rawBoxes.push_back(rect);
    }

    std::vector<cv::Rect> merged = mergeNearbyBoxes(rawBoxes, image.size());
    if (merged.empty()) {
        merged.push_back(cv::Rect(0, image.rows / 3, image.cols, std::max(1, image.rows / 3)));
        merged.push_back(cv::Rect(0, image.rows / 2, image.cols, std::max(1, image.rows / 3)));
    }

    for (const auto& rect : merged) {
        const cv::Rect bounded = rect & cv::Rect(0, 0, image.cols, image.rows);
        const double aspect = static_cast<double>(bounded.width) / std::max(1, bounded.height);
        const cv::Mat roiGray = gray(bounded);
        const double quality = computeImageQualityScore(roiGray);
        const double geometry = clamp01((std::min(aspect, 10.0) / 10.0) * 0.6 +
                                        std::min(static_cast<double>(bounded.area()) / imageArea * 8.0, 0.4));
        output.push_back({bounded, geometry, quality});
    }

    std::sort(output.begin(), output.end(), [](const CandidateRoi& a, const CandidateRoi& b) {
        return (a.geometryScore + a.imageQualityScore) > (b.geometryScore + b.imageQualityScore);
    });

    if (output.size() > 8) {
        output.resize(8);
    }

    if (timings != nullptr) {
        timings->push_back({"roi_filter_merge_ms", elapsedMs(filterStart, Clock::now())});
    }

    if (debugImages != nullptr) {
        debugImages->gray = gray;
        debugImages->clahe = clahe;
        debugImages->denoised = denoised;
        debugImages->absGradX = absGradX;
        debugImages->morph = morph;
        debugImages->threshold = thresh;
    }

    return output;
}

std::vector<cv::Rect> ImagePreprocessor::mergeNearbyBoxes(const std::vector<cv::Rect>& boxes,
                                                          const cv::Size& imageSize) {
    if (boxes.empty()) {
        return {};
    }

    std::vector<cv::Rect> merged = boxes;
    bool changed = true;
    while (changed) {
        changed = false;
        std::vector<bool> used(merged.size(), false);
        std::vector<cv::Rect> next;

        for (std::size_t i = 0; i < merged.size(); ++i) {
            if (used[i]) {
                continue;
            }
            cv::Rect current = merged[i];
            used[i] = true;

            for (std::size_t j = i + 1; j < merged.size(); ++j) {
                if (used[j]) {
                    continue;
                }
                const cv::Rect other = merged[j];
                const int horizontalGap = std::max(0, std::max(current.x, other.x) -
                                                      std::min(current.x + current.width, other.x + other.width));
                const int verticalGap = std::max(0, std::max(current.y, other.y) -
                                                    std::min(current.y + current.height, other.y + other.height));
                const bool verticallyAligned =
                    std::abs((current.y + current.height / 2) - (other.y + other.height / 2)) <
                    std::max(current.height, other.height);
                if ((horizontalGap < imageSize.width / 40 && verticallyAligned) ||
                    (verticalGap < imageSize.height / 60 && horizontalGap < imageSize.width / 12)) {
                    current |= other;
                    used[j] = true;
                    changed = true;
                }
            }
            next.push_back(current);
        }

        merged.swap(next);
    }

    return merged;
}

}  // namespace tyre
