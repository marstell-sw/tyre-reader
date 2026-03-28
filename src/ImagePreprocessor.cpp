#include "ImagePreprocessor.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace tyre {

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

std::vector<CandidateRoi> ImagePreprocessor::proposeTextRegions(const cv::Mat& image) const {
    std::vector<CandidateRoi> output;
    if (image.empty()) {
        return output;
    }

    const cv::Mat gray = toGrayscale(image);
    const cv::Mat clahe = applyClahe(gray);
    const cv::Mat denoised = bilateralDenoise(clahe);

    cv::Mat gradX;
    cv::Sobel(denoised, gradX, CV_32F, 1, 0, 3);
    cv::Mat absGradX;
    cv::convertScaleAbs(gradX, absGradX);

    cv::Mat morph = morphologyClose(absGradX, std::max(15, image.cols / 25), std::max(3, image.rows / 120));
    cv::Mat thresh;
    cv::threshold(morph, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    thresh = morphologyClose(thresh, std::max(17, image.cols / 20), std::max(3, image.rows / 100));
    thresh = morphologyOpen(thresh, 3, 3);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

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
