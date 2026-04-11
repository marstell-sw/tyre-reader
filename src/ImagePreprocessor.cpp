#include "ImagePreprocessor.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>

namespace tyre {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

double sampleCircleStrength(const cv::Mat& edgeMagnitude,
                            const cv::Point2f& center,
                            double radius,
                            int samples,
                            double startAngleDeg = 0.0,
                            double endAngleDeg = 360.0) {
    if (edgeMagnitude.empty() || radius <= 1.0 || samples < 16) {
        return 0.0;
    }

    double total = 0.0;
    int count = 0;
    const double startRad = startAngleDeg * CV_PI / 180.0;
    const double endRad = endAngleDeg * CV_PI / 180.0;
    for (int i = 0; i < samples; ++i) {
        const double alpha = static_cast<double>(i) / static_cast<double>(samples - 1);
        const double theta = startRad + (endRad - startRad) * alpha;
        const int x = cvRound(center.x + radius * std::cos(theta));
        const int y = cvRound(center.y + radius * std::sin(theta));
        if (x < 1 || y < 1 || x >= edgeMagnitude.cols - 1 || y >= edgeMagnitude.rows - 1) {
            continue;
        }
        total += static_cast<double>(edgeMagnitude.at<std::uint8_t>(y, x));
        ++count;
    }

    if (count == 0) {
        return 0.0;
    }
    return total / static_cast<double>(count);
}

float refineOuterRadiusFromEdges(const cv::Mat& blurred,
                                 const cv::Point2f& center,
                                 float innerHoleRadius,
                                 int minDim) {
    if (blurred.empty() || innerHoleRadius <= 0.0F) {
        return 0.0F;
    }

    cv::Mat gradX;
    cv::Mat gradY;
    cv::Sobel(blurred, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(blurred, gradY, CV_32F, 0, 1, 3);
    cv::Mat magnitude;
    cv::magnitude(gradX, gradY, magnitude);
    cv::Mat edgeMagnitude;
    cv::convertScaleAbs(magnitude, edgeMagnitude);
    cv::GaussianBlur(edgeMagnitude, edgeMagnitude, cv::Size(5, 5), 0.0);

    const double minRadius = std::max(innerHoleRadius * 1.18F, static_cast<float>(minDim) * 0.28F);
    const double maxRadius = std::min(innerHoleRadius * 1.75F, static_cast<float>(minDim) * 0.56F);
    if (maxRadius <= minRadius + 4.0) {
        return innerHoleRadius / 0.72F;
    }

    double bestScore = -1.0;
    float bestRadius = innerHoleRadius / 0.72F;
    const int samples = 720;
    for (double radius = minRadius; radius <= maxRadius; radius += 2.0) {
        const double edgeScore = sampleCircleStrength(edgeMagnitude, center, radius, samples, 200.0, 340.0);
        const double normalizedRadius = radius / std::max(1, minDim);
        const double score = edgeScore + normalizedRadius * 8.0;
        if (score > bestScore) {
            bestScore = score;
            bestRadius = static_cast<float>(radius);
        }
    }

    return bestRadius;
}

cv::RotatedRect ellipseFromCircle(const cv::Point2f& center, float radius) {
    return cv::RotatedRect(center, cv::Size2f(radius * 2.0F, radius * 2.0F), 0.0F);
}

struct ComboCircle {
    cv::Point2f center;
    float radius = 0.0F;
    std::vector<cv::Point2f> anchorPoints;
};

ComboCircle fitCircleLeastSquares(const std::vector<cv::Point2f>& points) {
    ComboCircle result;
    if (points.size() < 3) {
        return result;
    }

    cv::Mat a(static_cast<int>(points.size()), 3, CV_64F);
    cv::Mat b(static_cast<int>(points.size()), 1, CV_64F);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        const double x = points[static_cast<std::size_t>(i)].x;
        const double y = points[static_cast<std::size_t>(i)].y;
        a.at<double>(i, 0) = x;
        a.at<double>(i, 1) = y;
        a.at<double>(i, 2) = 1.0;
        b.at<double>(i, 0) = x * x + y * y;
    }

    cv::Mat solution;
    cv::solve(a, b, solution, cv::DECOMP_SVD);
    const double d = solution.at<double>(0, 0);
    const double e = solution.at<double>(1, 0);
    const double f = solution.at<double>(2, 0);
    const double cx = d * 0.5;
    const double cy = e * 0.5;
    const double radius = std::sqrt(std::max(0.0, f + cx * cx + cy * cy));

    result.center = cv::Point2f(static_cast<float>(cx), static_cast<float>(cy));
    result.radius = static_cast<float>(radius);
    result.anchorPoints.assign(points.begin(), points.end());
    return result;
}

ComboCircle findInnerFromBorders(const cv::Mat& gray) {
    ComboCircle result;
    if (gray.empty()) {
        return result;
    }

    const int h = gray.rows;
    const int w = gray.cols;
    struct ScanSpec { int sx; int sy; int dx; int dy; };
    const std::array<ScanSpec, 4> scans{{
        {w / 2, 0, 0, 1},
        {w / 2, h - 1, 0, -1},
        {0, h / 2, 1, 0},
        {w - 1, h / 2, -1, 0},
    }};

    std::vector<cv::Point2f> points;
    for (const auto& scan : scans) {
        std::vector<float> values;
        std::vector<cv::Point2f> coords;
        int x = scan.sx;
        int y = scan.sy;
        while (x >= 0 && x < w && y >= 0 && y < h) {
            values.push_back(static_cast<float>(gray.at<std::uint8_t>(y, x)));
            coords.emplace_back(static_cast<float>(x), static_cast<float>(y));
            x += scan.dx;
            y += scan.dy;
        }

        if (values.size() < 50) {
            continue;
        }

        std::vector<float> smooth(values.size(), 0.0F);
        const int radius = 7;
        for (int i = 0; i < static_cast<int>(values.size()); ++i) {
            float acc = 0.0F;
            int count = 0;
            for (int k = -radius; k <= radius; ++k) {
                const int idx = i + k;
                if (idx < 0 || idx >= static_cast<int>(values.size())) {
                    continue;
                }
                acc += values[static_cast<std::size_t>(idx)];
                ++count;
            }
            smooth[static_cast<std::size_t>(i)] = count > 0 ? acc / static_cast<float>(count) : values[static_cast<std::size_t>(i)];
        }

        bool inTire = false;
        for (int i = 0; i < static_cast<int>(smooth.size()); ++i) {
            const float value = smooth[static_cast<std::size_t>(i)];
            if (value > 80.0F) {
                inTire = true;
            }
            if (inTire && value < 50.0F) {
                points.push_back(coords[static_cast<std::size_t>(i)]);
                break;
            }
        }
    }

    return fitCircleLeastSquares(points);
}

float findOuterRadialCombo(const cv::Mat& gray, int cx, int cy) {
    if (gray.empty()) {
        return 0.0F;
    }

    const int h = gray.rows;
    const int w = gray.cols;
    const int maxRadius = std::min({cx, cy, w - cx, h - cy}) - 10;
    if (maxRadius < 120) {
        return 0.0F;
    }

    std::vector<int> radii;
    for (int i = 0; i < 360; ++i) {
        const double angle = 2.0 * CV_PI * static_cast<double>(i) / 360.0;
        std::vector<int> rr;
        std::vector<float> values;
        for (int radius = 10; radius < maxRadius; radius += 2) {
            const int x = cvRound(static_cast<double>(cx) + static_cast<double>(radius) * std::cos(angle));
            const int y = cvRound(static_cast<double>(cy) + static_cast<double>(radius) * std::sin(angle));
            if (x < 0 || x >= w || y < 0 || y >= h) {
                continue;
            }
            rr.push_back(radius);
            values.push_back(static_cast<float>(gray.at<std::uint8_t>(y, x)));
        }
        if (values.size() < 20) {
            continue;
        }

        std::vector<float> smooth(values.size(), 0.0F);
        const int kernelRadius = 3;
        for (int j = 0; j < static_cast<int>(values.size()); ++j) {
            float acc = 0.0F;
            int count = 0;
            for (int k = -kernelRadius; k <= kernelRadius; ++k) {
                const int idx = j + k;
                if (idx < 0 || idx >= static_cast<int>(values.size())) {
                    continue;
                }
                acc += values[static_cast<std::size_t>(idx)];
                ++count;
            }
            smooth[static_cast<std::size_t>(j)] = count > 0 ? acc / static_cast<float>(count) : values[static_cast<std::size_t>(j)];
        }

        for (int j = static_cast<int>(smooth.size()) - 2; j > 0; --j) {
            const float grad = smooth[static_cast<std::size_t>(j + 1)] - smooth[static_cast<std::size_t>(j)];
            if (grad < -10.0F && rr[static_cast<std::size_t>(j)] > 100) {
                radii.push_back(rr[static_cast<std::size_t>(j)]);
                break;
            }
        }
    }

    if (radii.empty()) {
        return 0.0F;
    }
    const auto mid = radii.begin() + static_cast<std::ptrdiff_t>(radii.size() / 2);
    std::nth_element(radii.begin(), mid, radii.end());
    return static_cast<float>(*mid);
}

cv::RotatedRect fitBestOuterEllipse(const std::vector<std::vector<cv::Point>>& contours,
                                    const cv::Point2f& center,
                                    float targetRadius,
                                    float holeRadius,
                                    int minDim) {
    double bestScore = -1.0;
    cv::RotatedRect bestEllipse = ellipseFromCircle(center, targetRadius);

    for (const auto& contour : contours) {
        if (contour.size() < 5) {
            continue;
        }
        const double area = cv::contourArea(contour);
        if (area < minDim * minDim * 0.035) {
            continue;
        }

        const cv::RotatedRect ellipse = cv::fitEllipse(contour);
        const double semiMajor = std::max(ellipse.size.width, ellipse.size.height) * 0.5;
        const double semiMinor = std::min(ellipse.size.width, ellipse.size.height) * 0.5;
        if (semiMinor < holeRadius * 1.05 || semiMajor > minDim * 0.68) {
            continue;
        }

        const double centerDistance = cv::norm(ellipse.center - center) / std::max(1.0, static_cast<double>(minDim));
        const double meanRadius = 0.5 * (semiMajor + semiMinor);
        const double radiusDistance = std::abs(meanRadius - targetRadius) / std::max(10.0, static_cast<double>(targetRadius));
        const double eccentricityPenalty = std::abs(semiMajor - semiMinor) / std::max(1.0, semiMajor);
        const double score = 2.4 - centerDistance * 3.5 - radiusDistance * 2.0 - eccentricityPenalty * 0.4;
        if (score > bestScore) {
            bestScore = score;
            bestEllipse = ellipse;
        }
    }

    return bestEllipse;
}

bool fitWheelFromIsolatedForeground(const cv::Mat& blurred,
                                    const cv::Point2f& imageCenter,
                                    cv::Point2f& outCenter,
                                    float& outHoleRadius,
                                    float& outOuterRadius,
                                    cv::RotatedRect& outInnerEllipse,
                                    cv::RotatedRect& outOuterEllipse,
                                    cv::Rect& outRect) {
    if (blurred.empty()) {
        return false;
    }

    cv::Mat objectMask;
    cv::threshold(blurred, objectMask, 245, 255, cv::THRESH_BINARY_INV);
    cv::morphologyEx(objectMask,
                     objectMask,
                     cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size(std::max(7, blurred.cols / 80),
                                                        std::max(7, blurred.rows / 80))));
    cv::morphologyEx(objectMask,
                     objectMask,
                     cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size(std::max(3, blurred.cols / 180),
                                                        std::max(3, blurred.rows / 180))));

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(objectMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return false;
    }

    double bestOuterScore = -1.0;
    std::vector<cv::Point> bestOuterContour;
    cv::RotatedRect bestOuter;
    for (const auto& contour : contours) {
        if (contour.size() < 5) {
            continue;
        }
        const double area = cv::contourArea(contour);
        if (area < blurred.cols * blurred.rows * 0.12) {
            continue;
        }
        const cv::RotatedRect ellipse = cv::fitEllipse(contour);
        const double centerDistance = cv::norm(ellipse.center - imageCenter) /
                                      std::max(1.0, static_cast<double>(std::min(blurred.cols, blurred.rows)));
        const double score = area / std::max(1.0, static_cast<double>(blurred.cols * blurred.rows)) - centerDistance * 0.4;
        if (score > bestOuterScore) {
            bestOuterScore = score;
            bestOuterContour = contour;
            bestOuter = ellipse;
        }
    }
    if (bestOuterScore < 0.0) {
        return false;
    }

    cv::Mat holeMask;
    cv::threshold(blurred, holeMask, 245, 255, cv::THRESH_BINARY);
    cv::Mat innerRegionMask = cv::Mat::zeros(blurred.size(), CV_8UC1);
    cv::ellipse(innerRegionMask,
                bestOuter,
                cv::Scalar(255),
                cv::FILLED,
                cv::LINE_AA);
    cv::bitwise_and(holeMask, innerRegionMask, holeMask);
    cv::morphologyEx(holeMask,
                     holeMask,
                     cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size(std::max(3, blurred.cols / 180),
                                                        std::max(3, blurred.rows / 180))));
    cv::morphologyEx(holeMask,
                     holeMask,
                     cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE,
                                               cv::Size(std::max(5, blurred.cols / 120),
                                                        std::max(5, blurred.rows / 120))));

    std::vector<std::vector<cv::Point>> holeContours;
    cv::findContours(holeMask, holeContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    double bestInnerScore = -1.0;
    cv::RotatedRect bestInner;
    for (const auto& contour : holeContours) {
        if (contour.size() < 5) {
            continue;
        }
        const double area = cv::contourArea(contour);
        if (area < bestOuter.size.area() * 0.08 || area > bestOuter.size.area() * 0.8) {
            continue;
        }
        const cv::RotatedRect ellipse = cv::fitEllipse(contour);
        const double centerDistance = cv::norm(ellipse.center - bestOuter.center) /
                                      std::max(1.0, static_cast<double>(std::min(blurred.cols, blurred.rows)));
        const double score = area / std::max(1.0, static_cast<double>(bestOuter.size.area())) - centerDistance * 0.6;
        if (score > bestInnerScore) {
            bestInnerScore = score;
            bestInner = ellipse;
        }
    }
    if (bestInnerScore < 0.0) {
        return false;
    }

    outOuterEllipse = bestOuter;
    outInnerEllipse = bestInner;
    outCenter = bestOuter.center;
    outOuterRadius = static_cast<float>(0.25 * (bestOuter.size.width + bestOuter.size.height));
    outHoleRadius = static_cast<float>(0.25 * (bestInner.size.width + bestInner.size.height));
    outRect = cv::boundingRect(bestOuterContour);
    return true;
}

bool estimateOuterCircleWithReducedHough(const cv::Mat& blurred,
                                         const cv::Point2f& preferredCenter,
                                         float preferredRadius,
                                         cv::Point2f& outCenter,
                                         float& outRadius,
                                         double& elapsedMsOut) {
    const Clock::time_point start = Clock::now();
    if (blurred.empty()) {
        elapsedMsOut = 0.0;
        return false;
    }

    const int maxDim = std::max(blurred.cols, blurred.rows);
    const double downscale = maxDim > 640 ? 640.0 / static_cast<double>(maxDim) : 1.0;
    cv::Mat small;
    if (downscale < 0.999) {
        cv::resize(blurred, small, cv::Size(), downscale, downscale, cv::INTER_AREA);
    } else {
        small = blurred;
    }

    const cv::Point2f scaledCenter(preferredCenter.x * static_cast<float>(downscale),
                                   preferredCenter.y * static_cast<float>(downscale));
    const float scaledRadius = preferredRadius * static_cast<float>(downscale);
    const int roiHalfSize = std::max(120, cvRound(scaledRadius * 1.18F));
    const int roiX = std::clamp(cvRound(scaledCenter.x) - roiHalfSize, 0, std::max(0, small.cols - 1));
    const int roiY = std::clamp(cvRound(scaledCenter.y) - roiHalfSize, 0, std::max(0, small.rows - 1));
    const int roiW = std::min(small.cols - roiX, std::max(1, roiHalfSize * 2));
    const int roiH = std::min(small.rows - roiY, std::max(1, roiHalfSize * 2));
    const cv::Rect roi(roiX, roiY, roiW, roiH);

    cv::Mat roiBlur = small(roi).clone();
    cv::GaussianBlur(roiBlur, roiBlur, cv::Size(7, 7), 0.0);

    std::vector<cv::Vec3f> circles;
    const int minRadius = std::max(20, cvRound(scaledRadius * 0.78F));
    const int maxRadius = std::max(minRadius + 5, cvRound(scaledRadius * 1.18F));
    cv::HoughCircles(roiBlur,
                     circles,
                     cv::HOUGH_GRADIENT,
                     1.2,
                     std::max(40.0, scaledRadius * 0.6),
                     100.0,
                     24.0,
                     minRadius,
                     maxRadius);

    elapsedMsOut = elapsedMs(start, Clock::now());
    if (circles.empty()) {
        return false;
    }

    double bestScore = -1.0;
    cv::Vec3f bestCircle = circles.front();
    for (const auto& circle : circles) {
        const cv::Point2f center(circle[0] + static_cast<float>(roi.x),
                                 circle[1] + static_cast<float>(roi.y));
        const double centerDistance = cv::norm(center - scaledCenter) / std::max(1.0F, scaledRadius);
        const double radiusDistance = std::abs(circle[2] - scaledRadius) / std::max(1.0F, scaledRadius);
        const double score = 2.0 - centerDistance * 1.5 - radiusDistance;
        if (score > bestScore) {
            bestScore = score;
            bestCircle = circle;
        }
    }

    outCenter = cv::Point2f((bestCircle[0] + static_cast<float>(roi.x)) / static_cast<float>(downscale),
                            (bestCircle[1] + static_cast<float>(roi.y)) / static_cast<float>(downscale));
    outRadius = bestCircle[2] / static_cast<float>(downscale);
    return true;
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

    const int minDim = std::min(blurred.cols, blurred.rows);

    const Clock::time_point contourMaskStart = Clock::now();
    cv::Mat darkMask;
    cv::threshold(blurred, darkMask, 110, 255, cv::THRESH_BINARY_INV);
    darkMask = morphologyClose(darkMask, std::max(9, blurred.cols / 40), std::max(9, blurred.rows / 40));
    darkMask = morphologyOpen(darkMask, std::max(5, blurred.cols / 120), std::max(5, blurred.rows / 120));
    if (timings != nullptr) {
        timings->push_back({"wheel_mask_ms", elapsedMs(contourMaskStart, Clock::now())});
    }

    if (debugImages != nullptr) {
        debugImages->gray = gray;
        debugImages->blurred = blurred;
        debugImages->darkMask = darkMask;
    }

    const cv::Point2f imageCenter(static_cast<float>(blurred.cols) * 0.5F,
                                  static_cast<float>(blurred.rows) * 0.5F);

    cv::Mat holeMask;
    cv::threshold(blurred, holeMask, 70, 255, cv::THRESH_BINARY_INV);
    holeMask = morphologyOpen(holeMask, std::max(3, blurred.cols / 180), std::max(3, blurred.rows / 180));
    holeMask = morphologyClose(holeMask, std::max(7, blurred.cols / 90), std::max(7, blurred.rows / 90));

    cv::Mat centerBiasMask = cv::Mat::zeros(blurred.size(), CV_8UC1);
    const cv::Size axes(std::max(8, static_cast<int>(std::round(blurred.cols * 0.34))),
                        std::max(8, static_cast<int>(std::round(blurred.rows * 0.34))));
    cv::ellipse(centerBiasMask,
                imageCenter,
                axes,
                0.0,
                0.0,
                360.0,
                cv::Scalar(255),
                cv::FILLED,
                cv::LINE_AA);
    cv::bitwise_and(holeMask, centerBiasMask, holeMask);

    const Clock::time_point contourStart = Clock::now();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(darkMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (timings != nullptr) {
        timings->push_back({"wheel_contours_ms", elapsedMs(contourStart, Clock::now())});
    }

    double bestScore = -1.0;
    cv::Point2f bestCenter;
    float bestRadius = 0.0F;
    float bestHoleRadius = 0.0F;
    cv::RotatedRect bestInnerEllipse;
    cv::Rect bestRect;
    int bestContourIndex = -1;

    std::vector<std::vector<cv::Point>> holeContours;
    cv::findContours(holeMask, holeContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const ComboCircle comboInner = findInnerFromBorders(blurred);
    if (comboInner.radius > minDim * 0.18F && comboInner.radius < minDim * 0.48F) {
        const float comboOuterRadius = findOuterRadialCombo(blurred, cvRound(comboInner.center.x), cvRound(comboInner.center.y));
        if (comboOuterRadius > comboInner.radius + 40.0F) {
            bestScore = 20.0;
            bestCenter = comboInner.center;
            bestHoleRadius = comboInner.radius;
            bestRadius = comboOuterRadius;
            bestInnerEllipse = ellipseFromCircle(comboInner.center, comboInner.radius);
            bestRect = cv::Rect(cvRound(bestCenter.x - bestRadius),
                                cvRound(bestCenter.y - bestRadius),
                                cvRound(bestRadius * 2.0F),
                                cvRound(bestRadius * 2.0F));
            bestContourIndex = -1;
        }
    }

    cv::Point2f isolatedCenter;
    float isolatedHoleRadius = 0.0F;
    float isolatedOuterRadius = 0.0F;
    cv::RotatedRect isolatedInnerEllipse;
    cv::RotatedRect isolatedOuterEllipse;
    cv::Rect isolatedRect;
    if (fitWheelFromIsolatedForeground(blurred,
                                       imageCenter,
                                       isolatedCenter,
                                       isolatedHoleRadius,
                                       isolatedOuterRadius,
                                       isolatedInnerEllipse,
                                       isolatedOuterEllipse,
                                       isolatedRect)) {
        bestScore = 10.0;
        bestCenter = isolatedCenter;
        bestHoleRadius = isolatedHoleRadius;
        bestRadius = isolatedOuterRadius;
        bestInnerEllipse = isolatedInnerEllipse;
        bestRect = isolatedRect;
        bestContourIndex = -1;
    }

    for (const auto& contour : holeContours) {
        const double area = cv::contourArea(contour);
        if (area < minDim * minDim * 0.015 || area > minDim * minDim * 0.24) {
            continue;
        }

        cv::Point2f center;
        float radius = 0.0F;
        cv::minEnclosingCircle(contour, center, radius);
        if (radius < minDim * 0.12F || radius > minDim * 0.42F) {
            continue;
        }

        const cv::Rect rect = cv::boundingRect(contour);
        const double aspect = static_cast<double>(rect.width) / std::max(1, rect.height);
        if (aspect < 0.75 || aspect > 1.25) {
            continue;
        }

        const double centerDistance = cv::norm(center - imageCenter) / std::max(1.0, static_cast<double>(minDim));
        const double fillRatio = area / std::max(1.0, CV_PI * radius * radius);
        const double normalizedRadius = radius / std::max(1.0, static_cast<double>(minDim));
        const double score = normalizedRadius * 1.4 + fillRatio * 1.6 - centerDistance * 1.8;
        if (score > bestScore) {
            const float refinedOuterRadius = refineOuterRadiusFromEdges(blurred, center, radius, minDim);
            bestScore = score;
            bestCenter = center;
            bestHoleRadius = radius;
            bestRadius = refinedOuterRadius > 0.0F ? refinedOuterRadius : radius / 0.72F;
            bestInnerEllipse = contour.size() >= 5 ? cv::fitEllipse(contour) : ellipseFromCircle(center, radius);
            bestRect = cv::Rect(cvRound(center.x - bestRadius), cvRound(center.y - bestRadius),
                                cvRound(bestRadius * 2.0F), cvRound(bestRadius * 2.0F));
            bestContourIndex = -1;
        }
    }

    for (std::size_t i = 0; i < contours.size(); ++i) {
        const double area = cv::contourArea(contours[i]);
        if (area < minDim * minDim * 0.05) {
            continue;
        }

        cv::Point2f center;
        float radius = 0.0F;
        cv::minEnclosingCircle(contours[i], center, radius);
        if (radius < minDim * 0.18F || radius > minDim * 0.60F) {
            continue;
        }

        const cv::Rect rect = cv::boundingRect(contours[i]);
        const double aspect = static_cast<double>(rect.width) / std::max(1, rect.height);
        if (aspect < 0.7 || aspect > 1.3) {
            continue;
        }

        const double centerDistance = cv::norm(center - imageCenter) / std::max(1.0, static_cast<double>(minDim));
        const double fillRatio = area / std::max(1.0, CV_PI * radius * radius);
        const double normalizedRadius = radius / std::max(1.0, static_cast<double>(minDim));
        const double score = normalizedRadius * 2.2 + fillRatio * 0.9 - centerDistance * 1.1;
        if (score > bestScore) {
            bestScore = score;
            bestCenter = center;
            bestRadius = radius;
            bestHoleRadius = radius * 0.68F;
            bestInnerEllipse = ellipseFromCircle(center, bestHoleRadius);
            bestRect = rect;
            bestContourIndex = static_cast<int>(i);
        }
    }

    std::vector<cv::Vec3f> circles;
    double reducedHoughMs = 0.0;
    const bool shouldTryReducedHough = false;
    if (shouldTryReducedHough) {
        cv::Point2f houghCenter = bestHoleRadius > 0.0F ? bestCenter : imageCenter;
        float houghRadius = bestRadius > 0.0F ? bestRadius : static_cast<float>(minDim) * 0.42F;
        cv::Point2f refinedCenter;
        float refinedRadius = 0.0F;
        if (estimateOuterCircleWithReducedHough(blurred, houghCenter, houghRadius, refinedCenter, refinedRadius, reducedHoughMs)) {
            const double centerDistance =
                cv::norm(refinedCenter - (bestHoleRadius > 0.0F ? bestCenter : imageCenter)) /
                std::max(1.0F, bestHoleRadius > 0.0F ? bestHoleRadius : refinedRadius);
            const double radiusDistance =
                std::abs(refinedRadius - (bestRadius > 0.0F ? bestRadius : refinedRadius)) /
                std::max(1.0F, bestRadius > 0.0F ? bestRadius : refinedRadius);
            const double score = 2.4 - centerDistance * 1.2 - radiusDistance * 0.8;
            if (score > bestScore - 0.1) {
                bestScore = score;
                bestCenter = refinedCenter;
                bestRadius = refinedRadius;
                if (bestHoleRadius <= 0.0F) {
                    bestHoleRadius = refinedRadius * 0.68F;
                }
                bestInnerEllipse = ellipseFromCircle(bestCenter, bestHoleRadius);
                bestRect = cv::Rect(cvRound(bestCenter.x - bestRadius), cvRound(bestCenter.y - bestRadius),
                                    cvRound(bestRadius * 2.0F), cvRound(bestRadius * 2.0F));
            }
        }
    }
    if (timings != nullptr) {
        timings->push_back({"wheel_hough_ms", reducedHoughMs});
    }

    if (bestScore < 0.0 || bestRadius <= 0.0F) {
        return geometry;
    }

    geometry.found = true;
    geometry.center = cv::Point2f(bestCenter.x / static_cast<float>(scale),
                                  bestCenter.y / static_cast<float>(scale));
    geometry.radius = bestRadius / static_cast<float>(scale);
    const float scaledHoleRadius = bestHoleRadius > 0.0F ? (bestHoleRadius / static_cast<float>(scale))
                                                          : (geometry.radius * 0.68F);
    geometry.innerRadius = scaledHoleRadius * 1.02F;
    geometry.outerRadius = geometry.radius * 0.98F;
    if (bestInnerEllipse.size.width > 0.0F && bestInnerEllipse.size.height > 0.0F) {
        geometry.innerEllipse = cv::RotatedRect(
            cv::Point2f(bestInnerEllipse.center.x / static_cast<float>(scale),
                        bestInnerEllipse.center.y / static_cast<float>(scale)),
            cv::Size2f(bestInnerEllipse.size.width / static_cast<float>(scale),
                       bestInnerEllipse.size.height / static_cast<float>(scale)),
            bestInnerEllipse.angle);
    } else {
        geometry.innerEllipse = ellipseFromCircle(geometry.center, geometry.innerRadius);
    }

    cv::RotatedRect bestOuterEllipseResized =
        fitBestOuterEllipse(contours, bestCenter, bestRadius, bestHoleRadius, minDim);
    if (isolatedOuterEllipse.size.width > 0.0F && isolatedOuterEllipse.size.height > 0.0F &&
        bestScore >= 9.5) {
        bestOuterEllipseResized = isolatedOuterEllipse;
    }
    geometry.outerEllipse = cv::RotatedRect(
        cv::Point2f(bestOuterEllipseResized.center.x / static_cast<float>(scale),
                    bestOuterEllipseResized.center.y / static_cast<float>(scale)),
        cv::Size2f(bestOuterEllipseResized.size.width / static_cast<float>(scale),
                   bestOuterEllipseResized.size.height / static_cast<float>(scale)),
        bestOuterEllipseResized.angle);
    geometry.center = geometry.outerEllipse.center;
    geometry.outerRadius = static_cast<float>(
        0.25 * (geometry.outerEllipse.size.width + geometry.outerEllipse.size.height));
    geometry.radius = geometry.outerRadius;
    geometry.bounds = cv::Rect(
        std::max(0, static_cast<int>(std::floor(geometry.center.x - geometry.outerRadius))),
        std::max(0, static_cast<int>(std::floor(geometry.center.y - geometry.outerRadius))),
        std::min(image.cols, static_cast<int>(std::ceil(geometry.outerRadius * 2.0F))),
        std::min(image.rows, static_cast<int>(std::ceil(geometry.outerRadius * 2.0F))));

    if (debugImages != nullptr) {
        cv::Mat contourOverlay;
        cv::cvtColor(resizedGray, contourOverlay, cv::COLOR_GRAY2BGR);
        if (bestContourIndex >= 0) {
            cv::drawContours(contourOverlay, contours, bestContourIndex, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }
        for (const auto& contour : holeContours) {
            cv::drawContours(contourOverlay, std::vector<std::vector<cv::Point>>{contour}, -1, cv::Scalar(255, 128, 0), 1, cv::LINE_AA);
        }
        cv::rectangle(contourOverlay, bestRect, cv::Scalar(255, 0, 255), 2, cv::LINE_AA);
        cv::circle(contourOverlay, bestCenter, cvRound(bestRadius), cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        if (bestInnerEllipse.size.width > 0.0F && bestInnerEllipse.size.height > 0.0F) {
            cv::ellipse(contourOverlay, bestInnerEllipse, cv::Scalar(255, 0, 0), 2, cv::LINE_AA);
        }
        debugImages->contourOverlay = contourOverlay;

        cv::Mat circlesOverlay;
        cv::cvtColor(resizedGray, circlesOverlay, cv::COLOR_GRAY2BGR);
        cv::circle(circlesOverlay,
                   bestCenter,
                   cvRound(bestRadius),
                   cv::Scalar(0, 255, 0),
                   3,
                   cv::LINE_AA);
        if (bestOuterEllipseResized.size.width > 0.0F && bestOuterEllipseResized.size.height > 0.0F) {
            cv::ellipse(circlesOverlay, bestOuterEllipseResized, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
        }
        debugImages->circlesOverlay = circlesOverlay;

        cv::Mat annulusOverlay = image.clone();
        cv::ellipse(annulusOverlay, geometry.outerEllipse, cv::Scalar(0, 255, 0), 3, cv::LINE_AA);
        cv::ellipse(annulusOverlay, geometry.innerEllipse, cv::Scalar(0, 140, 255), 3, cv::LINE_AA);
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
    const int angleSamples = std::max(1440, static_cast<int>(std::round(geometry.outerRadius * 2.0F * static_cast<float>(CV_PI))));
    const int radialSamples = std::max(128, static_cast<int>(std::ceil(geometry.outerRadius)));

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

    const double radialScale = static_cast<double>(radialSamples) / std::max(1.0F, geometry.outerRadius);
    const int innerIndex =
        std::clamp(static_cast<int>(std::floor(geometry.innerRadius * radialScale)), 0, std::max(0, polar.cols - 2));
    const int outerIndex =
        std::clamp(static_cast<int>(std::ceil(geometry.outerRadius * radialScale)), innerIndex + 1, polar.cols);
    cv::Mat sidewall = polar.colRange(innerIndex, outerIndex).clone();
    cv::transpose(sidewall, sidewall);
    cv::flip(sidewall, sidewall, 0);

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
