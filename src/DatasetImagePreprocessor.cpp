#include "DatasetImagePreprocessor.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

namespace tyre {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::string makeSafeStem(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        if (std::isalnum(c) != 0) {
            out.push_back(static_cast<char>(c));
        } else if (c == '_' || c == '-') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "image" : out;
}

}  // namespace

DatasetImagePreprocessor::ProcessedImageResult DatasetImagePreprocessor::processFile(const std::string& inputPath,
                                                                                    const std::string& outputDir) const {
    ProcessedImageResult result;
    result.inputPath = inputPath;

    const Clock::time_point totalStart = Clock::now();
    const Clock::time_point loadStart = Clock::now();
    const cv::Mat color = cv::imread(inputPath, cv::IMREAD_COLOR);
    result.stepTimings.push_back({"load_ms", elapsedMs(loadStart, Clock::now())});
    if (color.empty()) {
        result.stepTimings.push_back({"total_ms", elapsedMs(totalStart, Clock::now())});
        return result;
    }

    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);

    bool maskFound = false;
    cv::Mat mask = buildForegroundMask(gray, result.stepTimings, maskFound);
    result.maskFound = maskFound;

    const Clock::time_point applyStart = Clock::now();
    cv::Mat flattened(gray.size(), CV_8UC1, cv::Scalar(255));
    gray.copyTo(flattened, mask);
    result.stepTimings.push_back({"apply_mask_ms", elapsedMs(applyStart, Clock::now())});

    const Clock::time_point resizeStart = Clock::now();
    cv::Mat resized = resizeToTargetMegapixels(flattened, 5.0);
    result.stepTimings.push_back({"resize_5mp_ms", elapsedMs(resizeStart, Clock::now())});

    fs::create_directories(outputDir);
    const fs::path inputFsPath(inputPath);
    const std::string safeStem = makeSafeStem(inputFsPath.stem().string());
    result.outputPath = (fs::path(outputDir) / (safeStem + ".png")).string();

    const Clock::time_point saveStart = Clock::now();
    cv::imwrite(result.outputPath, resized);
    result.stepTimings.push_back({"save_ms", elapsedMs(saveStart, Clock::now())});
    result.stepTimings.push_back({"total_ms", elapsedMs(totalStart, Clock::now())});
    return result;
}

std::vector<DatasetImagePreprocessor::ProcessedImageResult> DatasetImagePreprocessor::processDirectory(
    const std::string& inputDir,
    const std::string& outputDir) const {
    std::vector<ProcessedImageResult> results;
    std::vector<fs::path> imagePaths;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        if (hasSupportedImageExtension(entry.path().extension().string())) {
            imagePaths.push_back(entry.path());
        }
    }

    std::sort(imagePaths.begin(), imagePaths.end());
    results.reserve(imagePaths.size());
    for (const auto& imagePath : imagePaths) {
        results.push_back(processFile(imagePath.string(), outputDir));
    }
    return results;
}

bool DatasetImagePreprocessor::hasSupportedImageExtension(const std::string& extension) {
    std::string ext;
    ext.reserve(extension.size());
    for (unsigned char c : extension) {
        if (std::isalnum(c) != 0) {
            ext.push_back(static_cast<char>(std::toupper(c)));
        }
    }
    return ext == "JPG" || ext == "JPEG" || ext == "PNG" || ext == "BMP" || ext == "TIFF" || ext == "TIF";
}

cv::Mat DatasetImagePreprocessor::buildForegroundMask(const cv::Mat& gray,
                                                      std::vector<NamedTiming>& timings,
                                                      bool& maskFound) {
    const Clock::time_point prepStart = Clock::now();
    const double scale = gray.cols > 1400 ? 1400.0 / static_cast<double>(gray.cols) : 1.0;
    cv::Mat smallGray;
    if (scale < 0.999) {
        cv::resize(gray, smallGray, cv::Size(), scale, scale, cv::INTER_AREA);
    } else {
        smallGray = gray;
    }

    cv::Mat blurred;
    cv::GaussianBlur(smallGray, blurred, cv::Size(9, 9), 0.0);
    cv::Mat thresholdMask;
    cv::threshold(blurred, thresholdMask, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
    const cv::Mat closeKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(std::max(11, smallGray.cols / 30), std::max(11, smallGray.rows / 30)));
    const cv::Mat openKernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE,
        cv::Size(std::max(5, smallGray.cols / 80), std::max(5, smallGray.rows / 80)));
    cv::morphologyEx(thresholdMask, thresholdMask, cv::MORPH_CLOSE, closeKernel);
    cv::morphologyEx(thresholdMask, thresholdMask, cv::MORPH_OPEN, openKernel);
    timings.push_back({"mask_prepare_ms", elapsedMs(prepStart, Clock::now())});

    const Clock::time_point contourStart = Clock::now();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresholdMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    timings.push_back({"mask_contours_ms", elapsedMs(contourStart, Clock::now())});

    const cv::Point2f center(static_cast<float>(smallGray.cols) * 0.5F,
                             static_cast<float>(smallGray.rows) * 0.5F);
    double bestScore = -1.0;
    int bestIndex = -1;
    for (std::size_t i = 0; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area < smallGray.cols * smallGray.rows * 0.05) {
            continue;
        }

        cv::Point2f contourCenter;
        float radius = 0.0F;
        cv::minEnclosingCircle(contours[i], contourCenter, radius);
        const double centerDistance =
            cv::norm(contourCenter - center) / std::max(1.0, static_cast<double>(std::min(smallGray.cols, smallGray.rows)));
        const double normalizedRadius = radius / std::max(1.0, static_cast<double>(std::min(smallGray.cols, smallGray.rows)));
        const double circularArea = CV_PI * radius * radius;
        const double fillRatio = area / std::max(1.0, circularArea);
        const double score = normalizedRadius * 2.0 + fillRatio - centerDistance;
        if (score > bestScore) {
            bestScore = score;
            bestIndex = static_cast<int>(i);
        }
    }

    cv::Mat smallMask = cv::Mat::zeros(smallGray.size(), CV_8UC1);
    if (bestIndex >= 0) {
        std::vector<cv::Point> hull;
        cv::convexHull(contours[bestIndex], hull);
        cv::fillConvexPoly(smallMask, hull, cv::Scalar(255));
        maskFound = true;
    } else {
        smallMask = cv::Mat(gray.size(), CV_8UC1, cv::Scalar(255));
        maskFound = false;
    }

    const Clock::time_point resizeStart = Clock::now();
    cv::Mat fullMask;
    if (scale < 0.999) {
        cv::resize(smallMask, fullMask, gray.size(), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
        fullMask = smallMask;
    }
    cv::threshold(fullMask, fullMask, 127, 255, cv::THRESH_BINARY);
    const cv::Mat refineKernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15));
    cv::morphologyEx(fullMask, fullMask, cv::MORPH_CLOSE, refineKernel);
    timings.push_back({"mask_resize_refine_ms", elapsedMs(resizeStart, Clock::now())});
    return fullMask;
}

cv::Mat DatasetImagePreprocessor::resizeToTargetMegapixels(const cv::Mat& image, double targetMegaPixels) {
    if (image.empty()) {
        return {};
    }

    const double currentPixels = static_cast<double>(image.cols) * static_cast<double>(image.rows);
    const double targetPixels = targetMegaPixels * 1000000.0;
    const double scale = std::sqrt(targetPixels / std::max(1.0, currentPixels));
    if (std::abs(scale - 1.0) < 0.02) {
        return image.clone();
    }

    cv::Mat resized;
    cv::resize(image,
               resized,
               cv::Size(std::max(1, static_cast<int>(std::round(image.cols * scale))),
                        std::max(1, static_cast<int>(std::round(image.rows * scale)))),
               0.0,
               0.0,
               scale > 1.0 ? cv::INTER_CUBIC : cv::INTER_AREA);
    return resized;
}

}  // namespace tyre
