#include "TyreAnalyzer.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <regex>
#include <sstream>
#include <thread>

#ifdef TYRE_READER_WITH_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace fs = std::filesystem;

namespace tyre {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

struct TextComponent {
    cv::Rect box;
    double centerY = 0.0;
};

double intersectionOverUnion(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect intersection = a & b;
    if (intersection.empty()) {
        return 0.0;
    }
    const double intersectionArea = static_cast<double>(intersection.area());
    const double unionArea = static_cast<double>(a.area() + b.area()) - intersectionArea;
    return unionArea > 0.0 ? intersectionArea / unionArea : 0.0;
}

std::vector<int> greedyNms(const std::vector<cv::Rect>& boxes,
                           const std::vector<float>& scores,
                           double iouThreshold) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int lhs, int rhs) {
        return scores[static_cast<std::size_t>(lhs)] > scores[static_cast<std::size_t>(rhs)];
    });

    std::vector<int> kept;
    std::vector<bool> suppressed(boxes.size(), false);
    for (std::size_t i = 0; i < order.size(); ++i) {
        const int currentIndex = order[i];
        if (suppressed[static_cast<std::size_t>(currentIndex)]) {
            continue;
        }
        kept.push_back(currentIndex);
        for (std::size_t j = i + 1; j < order.size(); ++j) {
            const int compareIndex = order[j];
            if (suppressed[static_cast<std::size_t>(compareIndex)]) {
                continue;
            }
            if (intersectionOverUnion(boxes[static_cast<std::size_t>(currentIndex)],
                                      boxes[static_cast<std::size_t>(compareIndex)]) > iouThreshold) {
                suppressed[static_cast<std::size_t>(compareIndex)] = true;
            }
        }
    }
    return kept;
}

bool isLikelyDotWeekYear(const std::string& weekYear) {
    if (weekYear.size() != 4) {
        return false;
    }
    const int week = std::stoi(weekYear.substr(0, 2));
    return week >= 1 && week <= 53;
}

double normalizeAngleDeg(double angleDeg) {
    while (angleDeg < 0.0) {
        angleDeg += 360.0;
    }
    while (angleDeg >= 360.0) {
        angleDeg -= 360.0;
    }
    return angleDeg;
}

cv::Mat cropUnwrappedSector(const cv::Mat& sidewall, double startAngleDeg, double endAngleDeg) {
    if (sidewall.empty()) {
        return {};
    }

    const double startNorm = normalizeAngleDeg(startAngleDeg);
    double endNorm = normalizeAngleDeg(endAngleDeg);
    if (std::abs(endNorm - startNorm) < 1e-6) {
        endNorm = startNorm + 360.0;
    } else if (endNorm <= startNorm) {
        endNorm += 360.0;
    }

    cv::Mat extended;
    cv::hconcat(sidewall, sidewall, extended);
    const double widthPerDegree = static_cast<double>(sidewall.cols) / 360.0;
    const int x0 = std::clamp(static_cast<int>(std::floor(startNorm * widthPerDegree)), 0, std::max(0, extended.cols - 1));
    const int x1 = std::clamp(static_cast<int>(std::ceil(endNorm * widthPerDegree)), x0 + 1, extended.cols);
    return extended(cv::Rect(x0, 0, x1 - x0, extended.rows)).clone();
}

struct LetterboxTransform {
    cv::Mat image;
    double scale = 1.0;
    int padX = 0;
    int padY = 0;
};

LetterboxTransform makeLetterboxedImage(const cv::Mat& image, int targetSize) {
    LetterboxTransform transform;
    if (image.empty()) {
        return transform;
    }

    const double scale = std::min(
        static_cast<double>(targetSize) / std::max(1, image.cols),
        static_cast<double>(targetSize) / std::max(1, image.rows));
    const int resizedWidth = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    const int resizedHeight = std::max(1, static_cast<int>(std::round(image.rows * scale)));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resizedWidth, resizedHeight), 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat canvas(targetSize, targetSize, CV_8UC3, cv::Scalar(114, 114, 114));
    const int padX = (targetSize - resizedWidth) / 2;
    const int padY = (targetSize - resizedHeight) / 2;
    resized.copyTo(canvas(cv::Rect(padX, padY, resizedWidth, resizedHeight)));

    transform.image = canvas;
    transform.scale = scale;
    transform.padX = padX;
    transform.padY = padY;
    return transform;
}

cv::Rect mapLetterboxedBoxToOriginal(double cx,
                                     double cy,
                                     double width,
                                     double height,
                                     const LetterboxTransform& transform,
                                     const cv::Size& originalSize) {
    const double x0 = (cx - width * 0.5 - transform.padX) / transform.scale;
    const double y0 = (cy - height * 0.5 - transform.padY) / transform.scale;
    const double x1 = (cx + width * 0.5 - transform.padX) / transform.scale;
    const double y1 = (cy + height * 0.5 - transform.padY) / transform.scale;

    const int ix0 = std::clamp(static_cast<int>(std::round(x0)), 0, std::max(0, originalSize.width - 1));
    const int iy0 = std::clamp(static_cast<int>(std::round(y0)), 0, std::max(0, originalSize.height - 1));
    const int ix1 = std::clamp(static_cast<int>(std::round(x1)), ix0 + 1, originalSize.width);
    const int iy1 = std::clamp(static_cast<int>(std::round(y1)), iy0 + 1, originalSize.height);
    return cv::Rect(ix0, iy0, ix1 - ix0, iy1 - iy0);
}

}  // namespace

#ifdef TYRE_READER_WITH_ONNXRUNTIME
struct TyreAnalyzer::YoloRuntime {
    Ort::Env env;
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> session;
    std::string modelPath;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
    std::vector<const char*> inputNamePtrs;
    std::vector<const char*> outputNamePtrs;

    YoloRuntime()
        : env(ORT_LOGGING_LEVEL_WARNING, "tyre_reader_yolo") {
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        const unsigned int hardwareThreads = std::max(1u, std::thread::hardware_concurrency());
        sessionOptions.SetIntraOpNumThreads(static_cast<int>(std::min(8u, hardwareThreads)));
        sessionOptions.SetInterOpNumThreads(1);
    }

    void load(const fs::path& path) {
        if (session && modelPath == path.string()) {
            return;
        }

        session = std::make_unique<Ort::Session>(env, path.string().c_str(), sessionOptions);
        modelPath = path.string();
        inputNames.clear();
        outputNames.clear();
        inputNamePtrs.clear();
        outputNamePtrs.clear();

        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t inputCount = session->GetInputCount();
        for (std::size_t i = 0; i < inputCount; ++i) {
            auto name = session->GetInputNameAllocated(i, allocator);
            inputNames.emplace_back(name.get());
        }
        const std::size_t outputCount = session->GetOutputCount();
        for (std::size_t i = 0; i < outputCount; ++i) {
            auto name = session->GetOutputNameAllocated(i, allocator);
            outputNames.emplace_back(name.get());
        }

        for (const auto& name : inputNames) {
            inputNamePtrs.push_back(name.c_str());
        }
        for (const auto& name : outputNames) {
            outputNamePtrs.push_back(name.c_str());
        }
    }
};
#endif

TyreAnalyzer::TyreAnalyzer(bool saveDebugArtifacts, bool skipOcr)
    : ocrEngine_(TesseractOcrEngine::Settings{}),
      saveDebugArtifacts_(saveDebugArtifacts),
      skipOcr_(skipOcr) {
}

TyreAnalyzer::~TyreAnalyzer() = default;

AnalysisResult TyreAnalyzer::analyzeImageFile(const std::string& imagePath, const std::string& outputDir) {
    AnalysisResult result;
    result.inputPath = imagePath;

    const Clock::time_point loadStart = Clock::now();
    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    appendTiming(result.stepTimings, "image_load_ms", elapsedMs(loadStart, Clock::now()));
    if (image.empty()) {
        result.notes.push_back("Unable to read input image.");
        return result;
    }

    result = analyzeFrame(image, fs::path(imagePath).stem().string(), outputDir);
    result.inputPath = imagePath;
    return result;
}

RoiOcrResult TyreAnalyzer::recognizeRoiFile(const std::string& imagePath,
                                            const cv::Rect& roi,
                                            const std::string& branch,
                                            const std::string& outputDir) {
    RoiOcrResult result;
    result.imagePath = imagePath;
    result.branch = branch;

    const Clock::time_point totalStart = Clock::now();
    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    appendTiming(result.stepTimings, "image_load_ms", elapsedMs(totalStart, Clock::now()));
    if (image.empty()) {
        result.notes.push_back("Unable to read input image.");
        return result;
    }

    const cv::Rect bounded = clampRect(roi, image.size());
    result.roi = bounded;
    if (bounded.empty()) {
        result.notes.push_back("Selected ROI is empty.");
        return result;
    }

    fs::create_directories(outputDir);
    const std::string safeStem = makeSafeStem(fs::path(imagePath).stem().string() + "_" + branch + "_roi");
    result.cropPath = (fs::path(outputDir) / (safeStem + ".png")).string();
    const cv::Mat crop = image(bounded).clone();
    cv::imwrite(result.cropPath, crop);

    if (!ocrEngine_.isInitialized()) {
        result.notes.push_back("Tesseract OCR initialization failed.");
        return result;
    }

    const bool isDot = normalizeForComparison(branch) == "DOT";
    const std::string whitelist = isDot ? "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -" : "0123456789R/ VWHYZ";

    const Clock::time_point variantStart = Clock::now();
    std::vector<std::pair<std::string, cv::Mat>> roiCandidates;
    if (!isDot && crop.rows >= 120) {
        const int h = crop.rows;
        const int w = crop.cols;
        const auto makeBand = [&](const std::string& name, double y0Ratio, double y1Ratio) {
            const int y0 = std::clamp(static_cast<int>(std::round(h * y0Ratio)), 0, std::max(0, h - 1));
            const int y1 = std::clamp(static_cast<int>(std::round(h * y1Ratio)), y0 + 1, h);
            roiCandidates.emplace_back(name, crop(cv::Rect(0, y0, w, y1 - y0)).clone());
        };
        makeBand("mid50", 0.18, 0.68);
        makeBand("top55", 0.00, 0.55);
        makeBand("top70", 0.00, 0.70);
        makeBand("upper_mid", 0.10, 0.72);
    }
    roiCandidates.emplace_back("full", crop);
    appendTiming(result.stepTimings, "variants_build_ms", elapsedMs(variantStart, Clock::now()));

    double bestScore = -1.0;
    bool shouldStop = false;
    for (const auto& roiCandidate : roiCandidates) {
        auto variants = buildFastVariants(roiCandidate.second, isDot);
        if (variants.size() > 4) {
            variants.resize(4);
        }
        for (const auto& variant : variants) {
            const std::vector<std::pair<std::string, tesseract::PageSegMode>> modes = {
                {roiCandidate.first + "_" + variant.first, tesseract::PSM_SINGLE_LINE},
                {roiCandidate.first + "_" + variant.first + "_raw", tesseract::PSM_RAW_LINE}
            };
            for (const auto& mode : modes) {
                const Clock::time_point ocrStart = Clock::now();
                const OcrResult ocr = ocrEngine_.recognize(variant.second, mode.first, mode.second, whitelist);
                appendTiming(result.stepTimings, "ocr_" + mode.first + "_ms", elapsedMs(ocrStart, Clock::now()));

                if (isDot) {
                    const ParsedDot parsed = parseDot(ocr.text);
                    const double completeness = parsed.fullFound ? 1.0 : (parsed.weekYearFound ? 0.78 : (parsed.dotFound ? 0.55 : 0.0));
                    const double score = clamp01(0.55 * ocr.averageConfidence + 0.45 * completeness);
                    if (score > bestScore) {
                        bestScore = score;
                        result.rawText = parsed.raw.empty() ? ocr.text : parsed.raw;
                        result.normalizedText = parsed.normalized.empty() ? parsed.fullNormalized : parsed.normalized;
                        result.found = parsed.dotFound || parsed.weekYearFound;
                        result.confidence = score;
                        if (parsed.fullFound && score >= 0.65) {
                            shouldStop = true;
                        }
                    }
                } else {
                    const ParsedSize parsed = parseTyreSize(ocr.text);
                    const double score = clamp01(0.55 * ocr.averageConfidence + 0.45 * parsed.parseQuality);
                    if (score > bestScore) {
                        bestScore = score;
                        result.rawText = ocr.text;
                        result.normalizedText = parsed.normalized;
                        result.found = parsed.found;
                        result.confidence = score;
                        result.notes = {"best_roi_candidate=" + roiCandidate.first};
                        if (parsed.found && score >= 0.65) {
                            shouldStop = true;
                        }
                    }
                }
                if (shouldStop) {
                    break;
                }
            }
            if (shouldStop) {
                break;
            }
        }
        if (shouldStop) {
            break;
        }
    }

    appendTiming(result.stepTimings, "total_ms", elapsedMs(totalStart, Clock::now()));
    return result;
}

SectorUnwrapResult TyreAnalyzer::unwrapSectorFile(const std::string& imagePath,
                                                  const std::string& branch,
                                                  double startAngleDeg,
                                                  double endAngleDeg,
                                                  bool useWheelOverride,
                                                  const cv::Point2f& wheelCenter,
                                                  float wheelInnerRadius,
                                                  float wheelOuterRadius,
                                                  const std::string& outputDir) {
    SectorUnwrapResult result;
    result.imagePath = imagePath;
    result.branch = branch;
    result.startAngleDeg = normalizeAngleDeg(startAngleDeg);
    result.endAngleDeg = normalizeAngleDeg(endAngleDeg);

    const Clock::time_point totalStart = Clock::now();
    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    appendTiming(result.stepTimings, "image_load_ms", elapsedMs(totalStart, Clock::now()));
    if (image.empty()) {
        result.notes.push_back("Unable to read input image.");
        return result;
    }

    fs::create_directories(outputDir);
    const std::string safeStem = makeSafeStem(fs::path(imagePath).stem().string() + "_" + branch + "_sector");
    result.overlayPath = (fs::path(outputDir) / (safeStem + "_overlay.png")).string();
    result.unwrappedPath = (fs::path(outputDir) / (safeStem + "_unwrap.png")).string();

    ImagePreprocessor::WheelDebugImages wheelDebugImages;
    const Clock::time_point wheelStart = Clock::now();
    ImagePreprocessor::WheelGeometry wheelGeometry;
    if (useWheelOverride) {
        wheelGeometry.found = wheelOuterRadius > wheelInnerRadius && wheelInnerRadius > 0.0F;
        wheelGeometry.center = wheelCenter;
        wheelGeometry.innerRadius = wheelInnerRadius;
        wheelGeometry.outerRadius = wheelOuterRadius;
        wheelGeometry.radius = wheelOuterRadius;
    } else {
        wheelGeometry = preprocessor_.detectWheelGeometry(image, &wheelDebugImages, &result.stepTimings);
    }
    appendTiming(result.stepTimings, "wheel_total_ms", elapsedMs(wheelStart, Clock::now()));
    result.wheelFound = wheelGeometry.found;
    result.wheelCenterX = wheelGeometry.center.x;
    result.wheelCenterY = wheelGeometry.center.y;
    result.wheelInnerRadius = wheelGeometry.innerRadius;
    result.wheelOuterRadius = wheelGeometry.outerRadius;

    cv::Mat overlay = image.clone();
    if (wheelGeometry.found) {
        cv::circle(overlay, wheelGeometry.center, cvRound(wheelGeometry.outerRadius), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        cv::circle(overlay, wheelGeometry.center, cvRound(wheelGeometry.innerRadius), cv::Scalar(0, 140, 255), 2, cv::LINE_AA);
        const auto endpointFor = [&](double angleDeg, float radius) {
            const double angleRad = angleDeg * CV_PI / 180.0;
            return cv::Point(
                cvRound(wheelGeometry.center.x + radius * std::cos(angleRad)),
                cvRound(wheelGeometry.center.y + radius * std::sin(angleRad)));
        };
        cv::line(overlay, endpointFor(result.startAngleDeg, wheelGeometry.innerRadius),
                 endpointFor(result.startAngleDeg, wheelGeometry.outerRadius), cv::Scalar(255, 80, 80), 3, cv::LINE_AA);
        cv::line(overlay, endpointFor(result.endAngleDeg, wheelGeometry.innerRadius),
                 endpointFor(result.endAngleDeg, wheelGeometry.outerRadius), cv::Scalar(80, 200, 255), 3, cv::LINE_AA);
    }
    cv::imwrite(result.overlayPath, overlay);
    appendTiming(result.stepTimings, "save_overlay_ms", elapsedMs(totalStart, Clock::now()));

    if (!wheelGeometry.found) {
        result.notes.push_back("Wheel not found.");
        return result;
    }

    const Clock::time_point unwrapStart = Clock::now();
    const cv::Mat sidewall = preprocessor_.unwrapSidewallBand(image, wheelGeometry, &wheelDebugImages, &result.stepTimings);
    const cv::Mat sector = cropUnwrappedSector(sidewall, result.startAngleDeg, result.endAngleDeg);
    appendTiming(result.stepTimings, "sector_unwrap_ms", elapsedMs(unwrapStart, Clock::now()));
    if (sector.empty()) {
        result.notes.push_back("Sector unwrap failed.");
        return result;
    }

    cv::imwrite(result.unwrappedPath, sector);
    appendTiming(result.stepTimings, "save_unwrapped_ms", elapsedMs(unwrapStart, Clock::now()));
    appendTiming(result.stepTimings, "total_ms", elapsedMs(totalStart, Clock::now()));
    return result;
}

std::vector<AnalysisResult> TyreAnalyzer::analyzeDirectory(const std::string& inputDir,
                                                           const std::string& outputDir) {
    std::vector<AnalysisResult> results;
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
    for (const auto& path : imagePaths) {
        results.push_back(analyzeImageFile(path.string(), outputDir));
    }

    return results;
}

WheelExtractionResult TyreAnalyzer::extractWheelGeometryFile(const std::string& imagePath, const std::string& outputDir) {
    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    WheelExtractionResult result;
    result.inputPath = imagePath;
    result.frameId = fs::path(imagePath).stem().string();
    if (image.empty()) {
        result.notes.push_back("Unable to read input image.");
        return result;
    }
    return extractWheelGeometryFrame(image, result.frameId, imagePath, outputDir);
}

std::vector<WheelExtractionResult> TyreAnalyzer::extractWheelGeometryDirectory(const std::string& inputDir,
                                                                               const std::string& outputDir) {
    std::vector<WheelExtractionResult> results;
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
    for (const auto& path : imagePaths) {
        results.push_back(extractWheelGeometryFile(path.string(), outputDir));
    }
    return results;
}

AnalysisResult TyreAnalyzer::analyzeFrame(const cv::Mat& frame,
                                          const std::string& frameId,
                                          const std::string& outputDir) {
    AnalysisResult result;
    result.frameId = frameId;

    const Clock::time_point totalStart = Clock::now();
    fs::create_directories(outputDir);

    const std::string safeStem = makeSafeStem(frameId.empty() ? "frame" : frameId);
    const std::string debugDir = (fs::path(outputDir) / safeStem / "debug").string();
    const std::string sizeCropPath = (fs::path(outputDir) / (safeStem + "_size_crop.png")).string();
    const std::string dotCropPath = (fs::path(outputDir) / (safeStem + "_dot_crop.png")).string();
    const std::string overlayPath = (fs::path(outputDir) / (safeStem + "_overlay.png")).string();
    result.debugDir = debugDir;
    result.timingReportPath = (fs::path(debugDir) / "timings.csv").string();
    result.ocrReportPath = (fs::path(debugDir) / "ocr_candidates.csv").string();
    if (saveDebugArtifacts_) {
        fs::create_directories(debugDir);
        saveDebugImage(frame, (fs::path(debugDir) / "00_input.png").string());
        std::ofstream ocrReport(result.ocrReportPath);
        ocrReport << "branch,roi_id,variant,psm,width,height,elapsed_ms,ocr_conf,regex_valid,domain_score,final_score,status,text,normalized\n";
    }

    const Clock::time_point grayStart = Clock::now();
    const cv::Mat gray = preprocessor_.toGrayscale(frame);
    appendTiming(result.stepTimings, "preprocess_gray_ms", elapsedMs(grayStart, Clock::now()));
    if (saveDebugArtifacts_) {
        saveDebugImage(gray, (fs::path(debugDir) / "01_gray.png").string());
    }

    const Clock::time_point claheStart = Clock::now();
    const cv::Mat grayClahe = preprocessor_.applyClahe(gray);
    appendTiming(result.stepTimings, "preprocess_clahe_ms", elapsedMs(claheStart, Clock::now()));
    if (saveDebugArtifacts_) {
        saveDebugImage(grayClahe, (fs::path(debugDir) / "02_gray_clahe.png").string());
    }

    const Clock::time_point wheelStart = Clock::now();
    ImagePreprocessor::WheelDebugImages wheelDebugImages;
    const ImagePreprocessor::WheelGeometry wheelGeometry =
        preprocessor_.detectWheelGeometry(frame, saveDebugArtifacts_ ? &wheelDebugImages : nullptr, &result.stepTimings);
    appendTiming(result.stepTimings, "wheel_total_ms", elapsedMs(wheelStart, Clock::now()));
    result.wheelFound = wheelGeometry.found;
    result.wheelCenterX = wheelGeometry.center.x;
    result.wheelCenterY = wheelGeometry.center.y;
    result.wheelInnerRadius = wheelGeometry.innerRadius;
    result.wheelOuterRadius = wheelGeometry.outerRadius;
    if (saveDebugArtifacts_) {
        saveDebugImage(wheelDebugImages.gray, (fs::path(debugDir) / "03_wheel_gray.png").string());
        saveDebugImage(wheelDebugImages.blurred, (fs::path(debugDir) / "04_wheel_blurred.png").string());
        saveDebugImage(wheelDebugImages.darkMask, (fs::path(debugDir) / "05_wheel_mask.png").string());
        saveDebugImage(wheelDebugImages.contourOverlay, (fs::path(debugDir) / "06_wheel_contours.png").string());
        saveDebugImage(wheelDebugImages.circlesOverlay, (fs::path(debugDir) / "07_wheel_circles.png").string());
        saveDebugImage(wheelDebugImages.annulusOverlay, (fs::path(debugDir) / "08_wheel_annulus.png").string());
    }
    result.timings.preprocessMs = elapsedMs(grayStart, Clock::now());

    cv::Mat ocrSource = frame;
    cv::Mat sidewallBand;
    if (wheelGeometry.found) {
        const Clock::time_point unwrapStart = Clock::now();
        sidewallBand =
            preprocessor_.unwrapSidewallBand(frame, wheelGeometry, saveDebugArtifacts_ ? &wheelDebugImages : nullptr, &result.stepTimings);
        appendTiming(result.stepTimings, "wheel_unwrap_total_ms", elapsedMs(unwrapStart, Clock::now()));
        if (!sidewallBand.empty()) {
            ocrSource = sidewallBand;
            result.notes.push_back("Wheel geometry available for local annulus unwrap.");
        } else {
            result.notes.push_back("Wheel detected, but sidewall unwrap failed. Falling back to full image.");
        }
    } else {
        result.notes.push_back("Wheel circle not detected. Falling back to full image.");
    }

    std::vector<cv::Rect> overlayBoxes;
    bool sizeFromYolo = false;
    bool dotFromYolo = false;
    const Clock::time_point proposalStart = Clock::now();
    const YoloPredictionRun yoloRun = runYoloRoiDetector(frame, frameId, debugDir);
    const bool yoloAvailable = yoloRun.ok;
    appendTiming(result.stepTimings, "yolo_inference_ms", yoloRun.elapsedMs);
    result.yoloOverlayPath = yoloRun.overlayPath;
    result.yoloDetections = yoloRun.detections;
    result.notes.insert(result.notes.end(), yoloRun.notes.begin(), yoloRun.notes.end());

    cv::Rect bestSizeBox;
    double bestSizeScore = -1.0;
    cv::Rect bestDotBox;
    double bestDotScore = -1.0;

    for (auto& detection : result.yoloDetections) {
        detection.box = clampRect(detection.box, frame.size());
        if (detection.box.empty()) {
            continue;
        }
        const std::string label = normalizeForComparison(detection.label);
        double annulusScore = 1.0;
        if (wheelGeometry.found) {
            annulusScore = computeAnnulusCompatibility(detection.box, wheelGeometry);
        }
        const bool semanticMatch = label == "SIZE" || label == "DOT";
        detection.acceptedForOcr = semanticMatch && (!wheelGeometry.found || annulusScore >= 0.34);
        const double combinedScore = detection.confidence * (wheelGeometry.found ? annulusScore : 1.0);
        if (!detection.acceptedForOcr) {
            continue;
        }
        if (label == "SIZE" && combinedScore > bestSizeScore) {
            bestSizeScore = combinedScore;
            bestSizeBox = detection.box;
        } else if (label == "DOT" && combinedScore > bestDotScore) {
            bestDotScore = combinedScore;
            bestDotBox = detection.box;
        }
    }
    if (saveDebugArtifacts_) {
        cv::Mat yoloSelected = frame.clone();
        for (const auto& detection : result.yoloDetections) {
            const cv::Scalar color = normalizeForComparison(detection.label) == "SIZE"
                                         ? (detection.acceptedForOcr ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 120, 0))
                                         : (normalizeForComparison(detection.label) == "DOT"
                                                ? (detection.acceptedForOcr ? cv::Scalar(0, 140, 255) : cv::Scalar(0, 70, 140))
                                                : cv::Scalar(180, 180, 180));
            const int thickness = detection.acceptedForOcr ? 3 : 1;
            cv::rectangle(yoloSelected, detection.box, color, thickness, cv::LINE_AA);
            cv::putText(yoloSelected,
                        detection.label + " " + formatDouble(detection.confidence, 2),
                        detection.box.tl() + cv::Point(0, -6),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        color,
                        2,
                        cv::LINE_AA);
        }
        saveDebugImage(yoloSelected, (fs::path(debugDir) / "12_yolo_selected.png").string());
    }
    result.timings.roiProposalMs = elapsedMs(proposalStart, Clock::now());

    auto runYoloBranch = [&](const std::string& branch, const cv::Rect& box, FieldResult& field, bool& fieldFound) {
        if (box.empty()) {
            return false;
        }

        std::string sourcePath;
        cv::Mat sourceImage;
        if (wheelGeometry.found && !sidewallBand.empty()) {
            const AnnulusLocalRoi localRoi =
                extractLocalAnnulusRoi(sidewallBand, wheelGeometry, box, branch == "dot");
            if (!localRoi.image.empty()) {
                sourceImage = localRoi.image;
                sourcePath = (fs::path(debugDir) / ("5" + std::string(branch == "size" ? "0" : "1") + "_yolo_" + branch + "_local.png")).string();
                saveDebugImage(sourceImage, sourcePath);
            }
        }
        if (sourcePath.empty()) {
            sourceImage = frame(box).clone();
            sourcePath = (fs::path(debugDir) / ("5" + std::string(branch == "size" ? "2" : "3") + "_yolo_" + branch + "_crop.png")).string();
            saveDebugImage(sourceImage, sourcePath);
        }

        if (sourceImage.empty()) {
            return false;
        }

        field.cropPath = sourcePath;
        field.boundingBox = box;
        field.roiQuality = wheelGeometry.found ? computeAnnulusCompatibility(box, wheelGeometry) : 1.0;
        if (skipOcr_) {
            result.notes.push_back("YOLO ROI prepared for " + branch + " OCR.");
            return false;
        }

        const Clock::time_point branchStart = Clock::now();
        const RoiOcrResult roiOcr = recognizeRoiFile(
            sourcePath,
            cv::Rect(0, 0, sourceImage.cols, sourceImage.rows),
            branch,
            (fs::path(debugDir) / ("yolo_" + branch + "_ocr")).string());
        appendTiming(result.stepTimings, "yolo_" + branch + "_total_ms", elapsedMs(branchStart, Clock::now()));
        for (const auto& timing : roiOcr.stepTimings) {
            appendTiming(result.stepTimings, "yolo_" + branch + "_" + timing.name, timing.ms);
        }
        if (!roiOcr.found) {
            return false;
        }

        field.rawText = roiOcr.rawText;
        field.normalizedText = roiOcr.normalizedText;
        field.found = roiOcr.found;
        field.confidence = roiOcr.confidence;
        field.uncertainty = 1.0 - roiOcr.confidence;
        fieldFound = true;
        overlayBoxes.push_back(box);
        result.notes.push_back("YOLO ROI accepted for " + branch + " OCR.");
        return true;
    };

    const Clock::time_point sizeStart = Clock::now();
    sizeFromYolo = runYoloBranch("size", bestSizeBox, result.tyreSize, result.tyreSizeFound);
    if (!sizeFromYolo && !yoloAvailable) {
        result.tyreSize = detectTyreSizeField(ocrSource, debugDir, result);
        if (result.tyreSize.found) {
            result.tyreSizeFound = true;
            overlayBoxes.push_back(result.tyreSize.boundingBox);
        }
    }
    appendTiming(result.stepTimings, "size_branch_total_ms", elapsedMs(sizeStart, Clock::now()));

    const Clock::time_point dotStart = Clock::now();
    dotFromYolo = runYoloBranch("dot", bestDotBox, result.dot, result.dotFound);
    if (!dotFromYolo && !yoloAvailable) {
        result.dot = detectDotField(ocrSource, debugDir, result);
        if (result.dot.found) {
            result.dotFound = true;
            overlayBoxes.push_back(result.dot.boundingBox);
        }
    }
    appendTiming(result.stepTimings, "dot_branch_total_ms", elapsedMs(dotStart, Clock::now()));
    result.timings.ocrMs = elapsedMs(dotStart, Clock::now());

    if (skipOcr_) {
        result.notes.push_back("OCR skipped by request after wheel/YOLO/local-unwarp extraction.");
        saveCropOrPlaceholder(frame, bestSizeBox, sizeCropPath, "size ROI prepared", !bestSizeBox.empty());
        saveCropOrPlaceholder(frame, bestDotBox, dotCropPath, "dot ROI prepared", !bestDotBox.empty());
        saveOverlay(frame, overlayBoxes, bestSizeBox, bestDotBox, overlayPath);
        result.overlayPath = overlayPath;
        result.tyreSize.cropPath = result.tyreSize.cropPath.empty() ? sizeCropPath : result.tyreSize.cropPath;
        result.dot.cropPath = result.dot.cropPath.empty() ? dotCropPath : result.dot.cropPath;
        result.timings.cropSaveMs = 0.0;
        result.timings.overlaySaveMs = 0.0;
        result.timings.totalMs = elapsedMs(totalStart, Clock::now());
        appendTiming(result.stepTimings, "total_ms", result.timings.totalMs);
        if (saveDebugArtifacts_) {
            writeDebugReport(result.timingReportPath, result.stepTimings);
        }
        return result;
    }

    if (result.dot.found) {
        const ParsedDot parsedDot = parseDot(result.dot.rawText);
        result.dotKeywordFound = parsedDot.dotFound;
        result.dotCodeBodyFound = parsedDot.fullFound || (!parsedDot.fullNormalized.empty() && parsedDot.fullNormalized.size() > 3);
        result.dotWeekYearFound = parsedDot.weekYearFound;
        result.dotFullFound = parsedDot.fullFound;
        result.dotWeekYear = parsedDot.weekYear;
        result.dotFullRaw = parsedDot.fullRaw;
        result.dotFullNormalized = parsedDot.fullNormalized;
    }
    result.timings.parsingMs = 0.0;

    if (!result.tyreSizeFound) {
        result.notes.push_back("Tyre size not found in candidate ROIs.");
    }
    if (!result.dotFound) {
        result.notes.push_back("DOT code not found in candidate ROIs.");
    } else if (!result.dotFullFound) {
        result.notes.push_back("Only partial DOT information available.");
    }

    const Clock::time_point cropStart = Clock::now();
    saveCropOrPlaceholder(sizeFromYolo ? frame : ocrSource,
                          result.tyreSize.boundingBox,
                          sizeCropPath,
                          "SIZE ROI NOT FOUND",
                          result.tyreSizeFound);
    saveCropOrPlaceholder(dotFromYolo ? frame : ocrSource,
                          result.dot.boundingBox,
                          dotCropPath,
                          "DOT ROI NOT FOUND",
                          result.dotFound);
    result.tyreSize.cropPath = sizeCropPath;
    result.dot.cropPath = dotCropPath;
    result.timings.cropSaveMs = elapsedMs(cropStart, Clock::now());

    const Clock::time_point overlayStart = Clock::now();
    if (sizeFromYolo || dotFromYolo) {
        saveOverlay(frame,
                    overlayBoxes,
                    sizeFromYolo ? result.tyreSize.boundingBox : cv::Rect(),
                    dotFromYolo ? result.dot.boundingBox : cv::Rect(),
                    overlayPath);
    } else {
        saveOverlay(ocrSource, overlayBoxes, result.tyreSize.boundingBox, result.dot.boundingBox, overlayPath);
    }
    result.overlayPath = overlayPath;
    result.timings.overlaySaveMs = elapsedMs(overlayStart, Clock::now());

    result.timings.totalMs = elapsedMs(totalStart, Clock::now());
    appendTiming(result.stepTimings, "total_ms", result.timings.totalMs);
    if (saveDebugArtifacts_) {
        if (!yoloAvailable) {
            cv::Mat roiOverlay = ocrSource.clone();
            for (std::size_t index = 0; index < overlayBoxes.size(); ++index) {
                cv::rectangle(roiOverlay, overlayBoxes[index], cv::Scalar(255, 200, 0), 2);
                cv::putText(roiOverlay, std::to_string(index), overlayBoxes[index].tl() + cv::Point(0, -4),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 2, cv::LINE_AA);
            }
            saveDebugImage(roiOverlay, (fs::path(debugDir) / "20_roi_overlay.png").string());
        }
        writeDebugReport(result.timingReportPath, result.stepTimings);
    }
    return result;
}

WheelExtractionResult TyreAnalyzer::extractWheelGeometryFrame(const cv::Mat& frame,
                                                              const std::string& frameId,
                                                              const std::string& inputPath,
                                                              const std::string& outputDir) const {
    WheelExtractionResult result;
    result.inputPath = inputPath;
    result.frameId = frameId;
    fs::create_directories(outputDir);

    const std::string safeStem = makeSafeStem(frameId.empty() ? "frame" : frameId);
    result.originalCopyPath = (fs::path(outputDir) / (safeStem + "_01_original.png")).string();
    result.wheelOverlayPath = (fs::path(outputDir) / (safeStem + "_02_wheel.png")).string();
    result.unwrappedBandPath = (fs::path(outputDir) / (safeStem + "_03_unwrap.png")).string();

    const Clock::time_point totalStart = Clock::now();
    const Clock::time_point saveOriginalStart = Clock::now();
    cv::imwrite(result.originalCopyPath, frame);
    appendTiming(result.stepTimings, "save_original_ms", elapsedMs(saveOriginalStart, Clock::now()));

    ImagePreprocessor::WheelDebugImages wheelDebugImages;
    const Clock::time_point wheelStart = Clock::now();
    const ImagePreprocessor::WheelGeometry wheelGeometry =
        preprocessor_.detectWheelGeometry(frame, &wheelDebugImages, &result.stepTimings);
    appendTiming(result.stepTimings, "wheel_total_ms", elapsedMs(wheelStart, Clock::now()));
    result.wheelFound = wheelGeometry.found;
    result.wheelCenterX = wheelGeometry.center.x;
    result.wheelCenterY = wheelGeometry.center.y;
    result.wheelInnerRadius = wheelGeometry.innerRadius;
    result.wheelOuterRadius = wheelGeometry.outerRadius;

    const Clock::time_point saveOverlayStart = Clock::now();
    if (!wheelDebugImages.contourOverlay.empty()) {
        cv::imwrite(result.wheelOverlayPath, wheelDebugImages.contourOverlay);
    } else if (!wheelDebugImages.circlesOverlay.empty()) {
        cv::imwrite(result.wheelOverlayPath, wheelDebugImages.circlesOverlay);
    } else {
        cv::imwrite(result.wheelOverlayPath, frame);
    }
    appendTiming(result.stepTimings, "save_wheel_overlay_ms", elapsedMs(saveOverlayStart, Clock::now()));

    const Clock::time_point unwrapStart = Clock::now();
    const cv::Mat unwrapped = preprocessor_.unwrapSidewallBand(frame, wheelGeometry, &wheelDebugImages, &result.stepTimings);
    appendTiming(result.stepTimings, "unwrap_total_ms", elapsedMs(unwrapStart, Clock::now()));

    const Clock::time_point saveUnwrapStart = Clock::now();
    if (!unwrapped.empty()) {
        cv::imwrite(result.unwrappedBandPath, unwrapped);
    } else {
        cv::Mat placeholder(160, 640, CV_8UC3, cv::Scalar(20, 20, 20));
        cv::putText(placeholder, "UNWRAP FAILED", cv::Point(20, 88), cv::FONT_HERSHEY_SIMPLEX, 1.0,
                    cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
        cv::imwrite(result.unwrappedBandPath, placeholder);
    }
    appendTiming(result.stepTimings, "save_unwrap_ms", elapsedMs(saveUnwrapStart, Clock::now()));

    if (!wheelGeometry.found) {
        result.notes.push_back("Wheel not found.");
    }
    if (wheelGeometry.found && unwrapped.empty()) {
        result.notes.push_back("Wheel found but unwrap failed.");
    }

    appendTiming(result.stepTimings, "total_ms", elapsedMs(totalStart, Clock::now()));
    return result;
}

std::string TyreAnalyzer::makeSafeStem(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(c));
        } else if (c == '_' || c == '-') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('_');
        }
    }
    return out.empty() ? "frame" : out;
}

std::string TyreAnalyzer::squeezeSpaces(const std::string& value) {
    std::ostringstream oss;
    bool previousSpace = false;
    for (unsigned char c : value) {
        if (std::isspace(c)) {
            if (!previousSpace) {
                oss << ' ';
            }
            previousSpace = true;
        } else {
            oss << static_cast<char>(c);
            previousSpace = false;
        }
    }
    std::string out = oss.str();
    if (!out.empty() && out.front() == ' ') {
        out.erase(out.begin());
    }
    if (!out.empty() && out.back() == ' ') {
        out.pop_back();
    }
    return out;
}

std::string TyreAnalyzer::sanitizeForParsing(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        const char upper = static_cast<char>(std::toupper(c));
        switch (upper) {
            case 'O':
            case 'Q':
                out.push_back('0');
                break;
            case 'I':
            case 'L':
                out.push_back('1');
                break;
            case 'S':
                out.push_back('5');
                break;
            default:
                out.push_back(upper);
                break;
        }
    }
    return squeezeSpaces(out);
}

std::string TyreAnalyzer::normalizeDotToken(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        if (std::isalnum(c)) {
            out.push_back(static_cast<char>(std::toupper(c)));
        }
    }
    return out;
}

bool TyreAnalyzer::hasSupportedImageExtension(const std::string& extension) {
    const std::string ext = normalizeForComparison(extension);
    return ext == "JPG" || ext == "JPEG" || ext == "PNG" || ext == "BMP" || ext == "TIFF" || ext == "TIF";
}

void TyreAnalyzer::appendTiming(std::vector<NamedTiming>& timings, const std::string& name, double ms) {
    timings.push_back({name, ms});
}

void TyreAnalyzer::saveDebugImage(const cv::Mat& image, const std::string& path) {
    if (!image.empty()) {
        cv::imwrite(path, image);
    }
}

void TyreAnalyzer::writeDebugReport(const std::string& path, const std::vector<NamedTiming>& timings) {
    std::ofstream out(path);
    out << "step,ms\n";
    for (const auto& timing : timings) {
        out << sanitizeCsvField(timing.name) << "," << formatDouble(timing.ms) << "\n";
    }
}

std::string TyreAnalyzer::sanitizeCsvField(const std::string& value) {
    std::string out = "\"";
    for (char c : value) {
        if (c == '\"') {
            out += "\"\"";
        } else {
            out.push_back(c);
        }
    }
    out += "\"";
    return out;
}

cv::Rect TyreAnalyzer::clampRect(const cv::Rect& rect, const cv::Size& bounds) {
    return rect & cv::Rect(0, 0, bounds.width, bounds.height);
}

std::pair<double, double> TyreAnalyzer::computeAngleRangeDeg(const cv::Rect& rect, const cv::Point2f& center) {
    std::vector<double> angles;
    angles.reserve(5);
    const std::vector<cv::Point2f> points = {
        cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y)),
        cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y)),
        cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y + rect.height)),
        cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y + rect.height)),
        cv::Point2f(static_cast<float>(rect.x + rect.width / 2.0), static_cast<float>(rect.y + rect.height / 2.0))
    };
    for (const auto& point : points) {
        angles.push_back(normalizeAngleDeg(std::atan2(point.y - center.y, point.x - center.x) * 180.0 / CV_PI));
    }
    std::sort(angles.begin(), angles.end());
    double largestGap = -1.0;
    std::size_t gapIndex = 0;
    for (std::size_t i = 0; i < angles.size(); ++i) {
        const double current = angles[i];
        const double next = (i + 1 < angles.size()) ? angles[i + 1] : (angles.front() + 360.0);
        const double gap = next - current;
        if (gap > largestGap) {
            largestGap = gap;
            gapIndex = i;
        }
    }

    double start = angles[(gapIndex + 1) % angles.size()];
    double end = angles[gapIndex];
    if (end <= start) {
        end += 360.0;
    }
    start -= 6.0;
    end += 6.0;
    return {normalizeAngleDeg(start), normalizeAngleDeg(end)};
}

double TyreAnalyzer::computeAnnulusCompatibility(const cv::Rect& rect, const ImagePreprocessor::WheelGeometry& geometry) {
    if (!geometry.found || geometry.outerRadius <= geometry.innerRadius) {
        return 1.0;
    }

    std::vector<double> radii;
    radii.reserve(5);
    const std::vector<cv::Point2f> points = {
        cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y)),
        cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y)),
        cv::Point2f(static_cast<float>(rect.x), static_cast<float>(rect.y + rect.height)),
        cv::Point2f(static_cast<float>(rect.x + rect.width), static_cast<float>(rect.y + rect.height)),
        cv::Point2f(static_cast<float>(rect.x + rect.width / 2.0), static_cast<float>(rect.y + rect.height / 2.0))
    };
    for (const auto& point : points) {
        radii.push_back(cv::norm(point - geometry.center));
    }
    const auto [minIt, maxIt] = std::minmax_element(radii.begin(), radii.end());
    const double boxMin = *minIt;
    const double boxMax = *maxIt;
    const double bandMin = geometry.innerRadius * 0.92;
    const double bandMax = geometry.outerRadius * 1.04;
    const double overlap = std::max(0.0, std::min(boxMax, bandMax) - std::max(boxMin, bandMin));
    const double boxSpan = std::max(1.0, boxMax - boxMin);
    const double overlapScore = clamp01(overlap / boxSpan);
    const double centerRadius = radii.back();
    const double normalizedCenter = (centerRadius - geometry.innerRadius) / std::max(1.0f, geometry.outerRadius - geometry.innerRadius);
    const double centerScore = clamp01(1.0 - std::abs(normalizedCenter - 0.52) / 0.52);
    return clamp01(0.65 * overlapScore + 0.35 * centerScore);
}

std::pair<double, double> TyreAnalyzer::computeRadiusRange(const cv::Rect& rect,
                                                           const cv::Point2f& center,
                                                           const ImagePreprocessor::WheelGeometry& geometry,
                                                           bool preferNarrowBand) {
    std::vector<double> radii;
    for (int gy = 0; gy <= 4; ++gy) {
        for (int gx = 0; gx <= 4; ++gx) {
            const double px = rect.x + rect.width * (static_cast<double>(gx) / 4.0);
            const double py = rect.y + rect.height * (static_cast<double>(gy) / 4.0);
            const double radius = cv::norm(cv::Point2f(static_cast<float>(px), static_cast<float>(py)) - center);
            if (radius >= geometry.innerRadius * 0.90 && radius <= geometry.outerRadius * 1.05) {
                radii.push_back(radius);
            }
        }
    }
    if (radii.empty()) {
        radii.push_back((geometry.innerRadius + geometry.outerRadius) * 0.5);
    }
    std::sort(radii.begin(), radii.end());
    const std::size_t n = radii.size();
    const double q25 = radii[n / 4];
    const double median = radii[n / 2];
    const double q75 = radii[(n * 3) / 4];
    const double bandWidth = std::max(1.0f, geometry.outerRadius - geometry.innerRadius);
    const double margin = preferNarrowBand ? bandWidth * 0.05 : bandWidth * 0.08;
    double radiusMin = std::max(static_cast<double>(geometry.innerRadius), q25 - margin);
    double radiusMax = std::min(static_cast<double>(geometry.outerRadius), q75 + margin);
    const double minSpan = preferNarrowBand ? bandWidth * 0.10 : bandWidth * 0.16;
    if (radiusMax - radiusMin < minSpan) {
        radiusMin = std::max(static_cast<double>(geometry.innerRadius), median - minSpan * 0.5);
        radiusMax = std::min(static_cast<double>(geometry.outerRadius), median + minSpan * 0.5);
    }
    return {radiusMin, radiusMax};
}

cv::Rect TyreAnalyzer::mapPolarWindowToBandRect(const cv::Size& sidewallBandSize,
                                                const ImagePreprocessor::WheelGeometry& geometry,
                                                double radiusMin,
                                                double radiusMax) {
    const double bandWidth = std::max(1.0f, geometry.outerRadius - geometry.innerRadius);
    const double normMin = clamp01((radiusMin - geometry.innerRadius) / bandWidth);
    const double normMax = clamp01((radiusMax - geometry.innerRadius) / bandWidth);
    int rowOuter = static_cast<int>(std::floor((1.0 - normMax) * sidewallBandSize.height));
    int rowInner = static_cast<int>(std::ceil((1.0 - normMin) * sidewallBandSize.height));
    rowOuter = std::clamp(rowOuter, 0, std::max(0, sidewallBandSize.height - 1));
    rowInner = std::clamp(rowInner, rowOuter + 1, sidewallBandSize.height);
    return cv::Rect(0, rowOuter, sidewallBandSize.width, rowInner - rowOuter);
}

TyreAnalyzer::AnnulusLocalRoi TyreAnalyzer::extractLocalAnnulusRoi(const cv::Mat& sidewallBand,
                                                                   const ImagePreprocessor::WheelGeometry& geometry,
                                                                   const cv::Rect& rect,
                                                                   bool preferNarrowBand) {
    AnnulusLocalRoi local;
    if (sidewallBand.empty() || !geometry.found) {
        return local;
    }
    auto angleRange = computeAngleRangeDeg(rect, geometry.center);
    auto radiusRange = computeRadiusRange(rect, geometry.center, geometry, preferNarrowBand);
    cv::Mat sector = cropUnwrappedSector(sidewallBand, angleRange.first, angleRange.second);
    if (sector.empty()) {
        return local;
    }
    const cv::Rect bandRect = mapPolarWindowToBandRect(sector.size(), geometry, radiusRange.first, radiusRange.second);
    const cv::Rect bounded = clampRect(bandRect, sector.size());
    if (bounded.empty()) {
        return local;
    }
    local.image = sector(bounded).clone();
    local.bandRect = bounded;
    local.startAngleDeg = angleRange.first;
    local.endAngleDeg = angleRange.second;
    local.radiusMin = radiusRange.first;
    local.radiusMax = radiusRange.second;
    return local;
}

TyreAnalyzer::YoloPredictionRun TyreAnalyzer::runYoloRoiDetector(const cv::Mat& image,
                                                                 const std::string& frameId,
                                                                 const std::string& debugDir) const {
    YoloPredictionRun run;
#ifndef TYRE_READER_WITH_ONNXRUNTIME
    run.notes.push_back("ONNX Runtime support is not available in this build. Falling back to heuristic ROI proposal.");
    return run;
#else
    const fs::path onnxPath = fs::path("ml_artifacts") / "yolo_runs" / "sidewall_v2_50ep_cpu" / "weights" / "best.onnx";
    if (!fs::exists(onnxPath)) {
        run.notes.push_back("YOLO ONNX model not found. Falling back to heuristic ROI proposal.");
        return run;
    }

    try {
        if (!yoloRuntime_) {
            yoloRuntime_ = std::make_unique<YoloRuntime>();
        }
        yoloRuntime_->load(onnxPath);
    } catch (const std::exception& ex) {
        run.notes.push_back("Failed to load YOLO ONNX model: " + std::string(ex.what()));
        return run;
    }

    const int inputSize = 640;
    const double confThreshold = 0.20;
    const double nmsThreshold = 0.45;
    const std::array<std::string, 4> classNames = {"Brand", "DOT", "Model", "Size"};

    const Clock::time_point inferStart = Clock::now();
    const LetterboxTransform letterbox = makeLetterboxedImage(image, inputSize);
    if (letterbox.image.empty()) {
        run.notes.push_back("Unable to prepare YOLO input image.");
        return run;
    }

    cv::Mat rgb;
    cv::cvtColor(letterbox.image, rgb, cv::COLOR_BGR2RGB);
    cv::Mat rgbFloat;
    rgb.convertTo(rgbFloat, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> channels;
    cv::split(rgbFloat, channels);
    std::vector<float> inputTensor(static_cast<std::size_t>(3 * inputSize * inputSize));
    const std::size_t planeSize = static_cast<std::size_t>(inputSize * inputSize);
    for (int channelIndex = 0; channelIndex < 3; ++channelIndex) {
        std::memcpy(inputTensor.data() + static_cast<std::size_t>(channelIndex) * planeSize,
                    channels[static_cast<std::size_t>(channelIndex)].ptr<float>(),
                    planeSize * sizeof(float));
    }

    const std::array<int64_t, 4> inputShape = {1, 3, inputSize, inputSize};
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value inputValue = Ort::Value::CreateTensor<float>(
        memoryInfo, inputTensor.data(), inputTensor.size(), inputShape.data(), inputShape.size());

    std::vector<Ort::Value> outputValues;
    try {
        outputValues = yoloRuntime_->session->Run(Ort::RunOptions{nullptr},
                                                  yoloRuntime_->inputNamePtrs.data(),
                                                  &inputValue,
                                                  1,
                                                  yoloRuntime_->outputNamePtrs.data(),
                                                  yoloRuntime_->outputNamePtrs.size());
    } catch (const std::exception& ex) {
        run.notes.push_back("YOLO ONNX inference failed: " + std::string(ex.what()));
        return run;
    }
    run.elapsedMs = elapsedMs(inferStart, Clock::now());

    if (outputValues.empty() || !outputValues.front().IsTensor()) {
        run.notes.push_back("YOLO ONNX inference returned no tensor outputs.");
        return run;
    }

    const Ort::Value& outputValue = outputValues.front();
    const auto outputShape = outputValue.GetTensorTypeAndShapeInfo().GetShape();
    if (outputShape.size() != 3 || outputShape[1] < 5 || outputShape[2] <= 0) {
        run.notes.push_back("Unexpected YOLO ONNX output shape.");
        return run;
    }

    const std::size_t channelCount = static_cast<std::size_t>(outputShape[1]);
    const std::size_t candidateCount = static_cast<std::size_t>(outputShape[2]);
    const std::size_t classCount = channelCount - 4;
    const float* predictions = outputValue.GetTensorData<float>();

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> classIds;
    boxes.reserve(candidateCount);
    scores.reserve(candidateCount);
    classIds.reserve(candidateCount);
    for (std::size_t candidateIndex = 0; candidateIndex < candidateCount; ++candidateIndex) {
        const float cx = predictions[candidateIndex];
        const float cy = predictions[candidateCount + candidateIndex];
        const float w = predictions[2 * candidateCount + candidateIndex];
        const float h = predictions[3 * candidateCount + candidateIndex];

        int bestClass = -1;
        float bestScore = 0.0F;
        for (std::size_t classIndex = 0; classIndex < classCount; ++classIndex) {
            const float score = predictions[(4 + classIndex) * candidateCount + candidateIndex];
            if (score > bestScore) {
                bestScore = score;
                bestClass = static_cast<int>(classIndex);
            }
        }
        if (bestClass < 0 || bestScore < confThreshold) {
            continue;
        }

        const cv::Rect mapped = mapLetterboxedBoxToOriginal(cx, cy, w, h, letterbox, image.size());
        if (mapped.width < 12 || mapped.height < 12) {
            continue;
        }
        boxes.push_back(mapped);
        scores.push_back(bestScore);
        classIds.push_back(bestClass);
    }

    std::vector<int> kept;
    for (int classIndex = 0; classIndex < static_cast<int>(classNames.size()); ++classIndex) {
        std::vector<cv::Rect> classBoxes;
        std::vector<float> classScores;
        std::vector<int> classOriginalIndices;
        for (std::size_t index = 0; index < boxes.size(); ++index) {
            if (classIds[index] != classIndex) {
                continue;
            }
            classBoxes.push_back(boxes[index]);
            classScores.push_back(scores[index]);
            classOriginalIndices.push_back(static_cast<int>(index));
        }
        const std::vector<int> classKept = greedyNms(classBoxes, classScores, nmsThreshold);
        for (int keptIndex : classKept) {
            kept.push_back(classOriginalIndices[static_cast<std::size_t>(keptIndex)]);
        }
    }
    std::sort(kept.begin(), kept.end(), [&](int lhs, int rhs) {
        return scores[static_cast<std::size_t>(lhs)] > scores[static_cast<std::size_t>(rhs)];
    });

    run.detections.reserve(kept.size());
    for (int index : kept) {
        if (classIds[static_cast<std::size_t>(index)] < 0 ||
            classIds[static_cast<std::size_t>(index)] >= static_cast<int>(classNames.size())) {
            continue;
        }
        YoloDetection detection;
        detection.label = classNames[static_cast<std::size_t>(classIds[static_cast<std::size_t>(index)])];
        detection.confidence = scores[static_cast<std::size_t>(index)];
        detection.box = boxes[static_cast<std::size_t>(index)];
        run.detections.push_back(std::move(detection));
    }

    fs::create_directories(debugDir);
    cv::Mat overlay = image.clone();
    for (const auto& detection : run.detections) {
        cv::Scalar color = cv::Scalar(0, 255, 255);
        const std::string labelUpper = normalizeForComparison(detection.label);
        if (labelUpper == "SIZE") {
            color = cv::Scalar(0, 255, 0);
        } else if (labelUpper == "DOT") {
            color = cv::Scalar(0, 140, 255);
        } else if (labelUpper == "BRAND") {
            color = cv::Scalar(255, 0, 255);
        } else if (labelUpper == "MODEL") {
            color = cv::Scalar(255, 255, 0);
        }
        cv::rectangle(overlay, detection.box, color, 3, cv::LINE_AA);
        cv::putText(overlay,
                    detection.label + " " + formatDouble(detection.confidence, 2),
                    detection.box.tl() + cv::Point(0, -6),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                    cv::LINE_AA);
    }
    const std::string safeStem = makeSafeStem(frameId.empty() ? "frame" : frameId);
    run.overlayPath = (fs::path(debugDir) / ("12_yolo_overlay_" + safeStem + ".png")).string();
    cv::imwrite(run.overlayPath, overlay);

    run.ok = !run.detections.empty();
    if (!run.ok) {
        run.notes.push_back("YOLO detector returned zero detections.");
    } else {
        run.notes.push_back("YOLO detector returned " + std::to_string(run.detections.size()) + " candidate ROI(s).");
    }
    return run;
#endif
}

std::vector<TyreAnalyzer::OcrProbe> TyreAnalyzer::buildStripProbes(const cv::Mat& image, const std::string& prefix) const {
    std::vector<OcrProbe> probes;
    if (image.empty()) {
        return probes;
    }

    auto addProbe = [&](const std::string& name, const cv::Rect& roi) {
        const cv::Rect bounded = roi & cv::Rect(0, 0, image.cols, image.rows);
        if (bounded.width < 80 || bounded.height < 20) {
            return;
        }
        probes.push_back({name, bounded, image(bounded).clone()});
    };

    addProbe(prefix + "_full", cv::Rect(0, 0, image.cols, image.rows));
    addProbe(prefix + "_mid", cv::Rect(0, image.rows / 4, image.cols, image.rows / 2));
    addProbe(prefix + "_top", cv::Rect(0, 0, image.cols, image.rows / 2));
    addProbe(prefix + "_bottom", cv::Rect(0, image.rows / 2, image.cols, image.rows / 2));
    addProbe(prefix + "_center_band", cv::Rect(0, image.rows / 3, image.cols, std::max(24, image.rows / 3)));

    if (prefix == "size") {
        const int windowWidth = std::max(260, image.cols / 3);
        const int stride = std::max(180, image.cols / 6);
        const int bandY = image.rows / 5;
        const int bandH = std::max(28, (image.rows * 3) / 5);
        for (int x = 0, index = 0; x < image.cols; x += stride, ++index) {
            addProbe(prefix + "_win_" + std::to_string(index), cv::Rect(x, bandY, windowWidth, bandH));
        }
    }

    if (prefix == "dot") {
        const int windowWidth = std::max(180, image.cols / 3);
        const int stride = std::max(120, image.cols / 5);
        const int bandY = image.rows / 4;
        const int bandH = std::max(28, image.rows / 2);
        for (int x = 0, index = 0; x < image.cols; x += stride, ++index) {
            addProbe(prefix + "_win_" + std::to_string(index), cv::Rect(x, bandY, windowWidth, bandH));
        }
    }

    return probes;
}

std::vector<std::pair<std::string, cv::Mat>> TyreAnalyzer::buildFastVariants(const cv::Mat& image,
                                                                              bool aggressiveThreshold) const {
    std::vector<std::pair<std::string, cv::Mat>> variants;
    if (image.empty()) {
        return variants;
    }

    cv::Mat gray = preprocessor_.toGrayscale(image);
    if (gray.rows < 180) {
        const double scale = std::max(2.0, 180.0 / std::max(1, gray.rows));
        gray = preprocessor_.resizeUpscale(gray, scale);
    } else if (gray.rows < 280) {
        gray = preprocessor_.resizeUpscale(gray, 1.5);
    }
    const cv::Mat clahe = preprocessor_.applyClahe(gray);
    variants.emplace_back("gray", gray);
    variants.emplace_back("clahe", clahe);
    cv::Mat blackhat;
    const cv::Mat blackhatKernel =
        cv::getStructuringElement(cv::MORPH_RECT, cv::Size(std::max(17, gray.cols / 14), std::max(5, gray.rows / 5)));
    cv::morphologyEx(clahe, blackhat, cv::MORPH_BLACKHAT, blackhatKernel);
    variants.emplace_back("blackhat", blackhat);
    cv::Mat otsu;
    cv::threshold(clahe, otsu, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    variants.emplace_back("otsu", otsu);
    const cv::Mat adaptive = preprocessor_.adaptiveThresholdImage(clahe);
    variants.emplace_back("adaptive", adaptive);
    variants.emplace_back("adaptive_inv", preprocessor_.invertImage(adaptive));
    if (aggressiveThreshold) {
        cv::Mat blackhatInv = preprocessor_.invertImage(blackhat);
        variants.emplace_back("blackhat_inv", blackhatInv);
    }
    return variants;
}

std::vector<TyreAnalyzer::StripCandidate> TyreAnalyzer::detectSidewallCandidates(const cv::Mat& stripImage,
                                                                                 const std::string& fieldName,
                                                                                 const std::string& debugDir,
                                                                                 AnalysisResult& result) const {
    std::vector<StripCandidate> candidates;
    if (stripImage.empty()) {
        return candidates;
    }

    const bool isCyclicStrip = stripImage.cols > stripImage.rows * 4;
    const int overlap = isCyclicStrip ? std::clamp(stripImage.cols / 6, 120, 640) : 0;
    cv::Mat extended = stripImage.clone();
    if (isCyclicStrip && overlap > 0) {
        std::vector<cv::Mat> tiles = {
            stripImage.colRange(stripImage.cols - overlap, stripImage.cols),
            stripImage,
            stripImage.colRange(0, overlap)
        };
        cv::hconcat(tiles, extended);
    }

    const Clock::time_point grayStart = Clock::now();
    const cv::Mat gray = preprocessor_.toGrayscale(extended);
    appendTiming(result.stepTimings, fieldName + "_candidate_gray_ms", elapsedMs(grayStart, Clock::now()));

    const Clock::time_point claheStart = Clock::now();
    const cv::Mat clahe = preprocessor_.applyClahe(gray);
    appendTiming(result.stepTimings, fieldName + "_candidate_clahe_ms", elapsedMs(claheStart, Clock::now()));

    const Clock::time_point blackhatStart = Clock::now();
    cv::Mat blackhat;
    const cv::Mat blackhatKernel = cv::getStructuringElement(
        cv::MORPH_RECT,
        fieldName == "size"
            ? cv::Size(std::max(13, extended.cols / 96), std::max(5, extended.rows / 8))
            : cv::Size(std::max(15, extended.cols / 48), std::max(5, extended.rows / 7)));
    cv::morphologyEx(clahe, blackhat, cv::MORPH_BLACKHAT, blackhatKernel);
    appendTiming(result.stepTimings, fieldName + "_candidate_blackhat_ms", elapsedMs(blackhatStart, Clock::now()));

    const Clock::time_point sobelStart = Clock::now();
    cv::Mat gradX;
    cv::Sobel(blackhat, gradX, CV_32F, 1, 0, 3);
    cv::Mat absGradX;
    cv::convertScaleAbs(gradX, absGradX);
    appendTiming(result.stepTimings, fieldName + "_candidate_sobel_ms", elapsedMs(sobelStart, Clock::now()));

    const Clock::time_point threshStart = Clock::now();
    cv::Mat thresh;
    cv::threshold(absGradX, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    if (fieldName == "size") {
        thresh = preprocessor_.morphologyOpen(thresh, 2, 2);
        thresh = preprocessor_.morphologyClose(
            thresh,
            std::max(3, extended.cols / 260),
            std::max(5, extended.rows / 7));
    } else {
        thresh = preprocessor_.morphologyClose(
            thresh,
            std::max(17, extended.cols / 28),
            std::max(5, extended.rows / 8));
        thresh = preprocessor_.morphologyOpen(thresh, 3, 3);
    }
    appendTiming(result.stepTimings, fieldName + "_candidate_threshold_ms", elapsedMs(threshStart, Clock::now()));

    const Clock::time_point contourStart = Clock::now();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    appendTiming(result.stepTimings, fieldName + "_candidate_contours_ms", elapsedMs(contourStart, Clock::now()));

    cv::Mat overlay;
    if (saveDebugArtifacts_) {
        cv::cvtColor(gray, overlay, cv::COLOR_GRAY2BGR);
        saveDebugImage(gray, (fs::path(debugDir) / ("50_" + fieldName + "_candidate_gray.png")).string());
        saveDebugImage(clahe, (fs::path(debugDir) / ("51_" + fieldName + "_candidate_clahe.png")).string());
        saveDebugImage(blackhat, (fs::path(debugDir) / ("52_" + fieldName + "_candidate_blackhat.png")).string());
        saveDebugImage(absGradX, (fs::path(debugDir) / ("53_" + fieldName + "_candidate_sobel.png")).string());
        saveDebugImage(thresh, (fs::path(debugDir) / ("54_" + fieldName + "_candidate_threshold.png")).string());
    }

    std::vector<TextComponent> components;
    const double minHeight = fieldName == "size" ? extended.rows * 0.16 : extended.rows * 0.06;
    const double maxHeight = fieldName == "size" ? extended.rows * 0.78 : extended.rows * 0.70;
    const double minWidth = fieldName == "size" ? 4.0 : 4.0;
    const double maxWidth = fieldName == "size" ? extended.cols * 0.035 : extended.cols * 0.12;

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    cv::connectedComponentsWithStats(thresh, labels, stats, centroids, 8, CV_32S);
    for (int label = 1; label < stats.rows; ++label) {
        const int x = stats.at<int>(label, cv::CC_STAT_LEFT);
        const int y = stats.at<int>(label, cv::CC_STAT_TOP);
        const int width = stats.at<int>(label, cv::CC_STAT_WIDTH);
        const int height = stats.at<int>(label, cv::CC_STAT_HEIGHT);
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (height < minHeight || height > maxHeight) {
            continue;
        }
        if (width < minWidth || width > maxWidth) {
            continue;
        }
        if (area < width * height * 0.15) {
            continue;
        }
        components.push_back({cv::Rect(x, y, width, height), y + height * 0.5});
    }

    std::sort(components.begin(), components.end(), [](const TextComponent& a, const TextComponent& b) {
        return a.box.x < b.box.x;
    });

    auto addCandidateFromRange = [&](std::size_t beginIndex, std::size_t endIndex, const std::string& tag) {
        if (beginIndex > endIndex || endIndex >= components.size()) {
            return;
        }
        cv::Rect rect = components[beginIndex].box;
        double avgHeight = 0.0;
        double baselineJitter = 0.0;
        std::vector<int> widths;
        std::vector<int> gaps;
        for (std::size_t i = beginIndex; i <= endIndex; ++i) {
            rect |= components[i].box;
            avgHeight += components[i].box.height;
            widths.push_back(components[i].box.width);
            if (i > beginIndex) {
                gaps.push_back(components[i].box.x - (components[i - 1].box.x + components[i - 1].box.width));
            }
        }
        const std::size_t count = endIndex - beginIndex + 1;
        avgHeight /= static_cast<double>(count);
        for (std::size_t i = beginIndex; i <= endIndex; ++i) {
            baselineJitter += std::abs(components[i].centerY - (rect.y + rect.height * 0.5));
        }
        baselineJitter /= static_cast<double>(count);

        const auto averageOf = [](const std::vector<int>& values) {
            double sum = 0.0;
            for (int value : values) {
                sum += static_cast<double>(value);
            }
            return values.empty() ? 0.0 : sum / static_cast<double>(values.size());
        };
        const auto stddevOf = [&](const std::vector<int>& values, double mean) {
            if (values.empty()) {
                return 0.0;
            }
            double acc = 0.0;
            for (int value : values) {
                const double delta = static_cast<double>(value) - mean;
                acc += delta * delta;
            }
            return std::sqrt(acc / static_cast<double>(values.size()));
        };
        const double meanWidth = averageOf(widths);
        const double meanGap = averageOf(gaps);
        const double gapStd = stddevOf(gaps, meanGap);

        rect.x -= std::max(12, rect.width / 12);
        rect.y -= std::max(8, static_cast<int>(avgHeight * 0.25));
        rect.width += std::max(24, rect.width / 6);
        rect.height += std::max(18, static_cast<int>(avgHeight * 0.45));
        rect = clampRect(rect, extended.size());

        const double aspect = static_cast<double>(rect.width) / std::max(1, rect.height);
        const double minAspect = fieldName == "size" ? 2.2 : 1.4;
        const double maxAspect = fieldName == "size" ? 12.0 : 10.0;
        if (aspect < minAspect || aspect > maxAspect) {
            return;
        }

        const int minChars = fieldName == "size" ? 4 : 3;
        const int maxChars = fieldName == "size" ? 14 : 16;
        if (static_cast<int>(count) < minChars || static_cast<int>(count) > maxChars) {
            return;
        }

        const int minWidthRoi = fieldName == "size" ? 160 : 110;
        const int maxWidthRoi = fieldName == "size" ? std::max(320, extended.cols / 3) : std::max(240, extended.cols / 4);
        if (rect.width < minWidthRoi || rect.width > maxWidthRoi) {
            return;
        }

        const cv::Rect mappedRect = isCyclicStrip
            ? cv::Rect((rect.x - overlap + stripImage.cols) % stripImage.cols,
                       rect.y,
                       std::min(rect.width, stripImage.cols),
                       rect.height)
            : rect;

        const cv::Mat crop = extended(rect).clone();
        const double roiQuality = preprocessor_.computeImageQualityScore(preprocessor_.toGrayscale(crop));
        const double countScore = fieldName == "size"
            ? 1.0 - std::min(std::abs(static_cast<double>(count) - 8.0) / 6.0, 1.0)
            : 1.0 - std::min(std::abs(static_cast<double>(count) - 6.0) / 8.0, 1.0);
        const double aspectScore = fieldName == "size"
            ? 1.0 - std::min(std::abs(aspect - 4.2) / 3.6, 1.0)
            : 1.0 - std::min(std::abs(aspect - 3.0) / 3.0, 1.0);
        const double heightScore = std::min(avgHeight / std::max(1.0, extended.rows * 0.35), 1.0);
        const double alignmentScore = 1.0 - std::min(baselineJitter / std::max(6.0, avgHeight * 0.35), 1.0);
        const double spacingScore = gaps.empty() ? 0.0 : 1.0 - std::min(gapStd / std::max(3.0, meanGap + 1.0), 1.0);
        int slenderCount = 0;
        int wideCount = 0;
        for (int width : widths) {
            const double widthRatio = width / std::max(1.0, avgHeight);
            if (widthRatio < 0.38) {
                ++slenderCount;
            }
            if (widthRatio > 0.55) {
                ++wideCount;
            }
        }
        const double slashAnchorScore = fieldName == "size" ? std::min(static_cast<double>(slenderCount) / 2.0, 1.0) : 0.0;
        const double digitClusterScore = std::min(static_cast<double>(wideCount) / 4.0, 1.0);
        const double structureScore = fieldName == "size"
            ? clamp01(0.24 * countScore + 0.18 * aspectScore + 0.16 * alignmentScore + 0.14 * spacingScore +
                      0.14 * slashAnchorScore + 0.06 * digitClusterScore + 0.08 * heightScore)
            : clamp01(0.32 * countScore + 0.24 * aspectScore + 0.16 * alignmentScore + 0.10 * spacingScore + 0.18 * roiQuality);
        const double score = clamp01(0.78 * structureScore + 0.22 * roiQuality);

        if (fieldName == "size" && structureScore < 0.46) {
            return;
        }

        StripCandidate candidate;
        candidate.name = fieldName + "_" + tag + "_" + std::to_string(candidates.size());
        candidate.extendedBox = rect;
        candidate.mappedBox = mappedRect;
        candidate.image = crop;
        candidate.geometryScore = score;
        candidate.imageQualityScore = roiQuality;
        candidates.push_back(std::move(candidate));
    };

    if (!components.empty()) {
        std::size_t runStart = 0;
        for (std::size_t i = 1; i <= components.size(); ++i) {
            bool split = (i == components.size());
            if (!split) {
                const int gap = components[i].box.x - (components[i - 1].box.x + components[i - 1].box.width);
                const double avgHeight = (components[i].box.height + components[i - 1].box.height) * 0.5;
                const double centerDelta = std::abs(components[i].centerY - components[i - 1].centerY);
                split = gap > std::max(18.0, avgHeight * (fieldName == "size" ? 1.1 : 1.4)) ||
                        centerDelta > std::max(18.0, avgHeight * 0.65);
            }
            if (split) {
                addCandidateFromRange(runStart, i - 1, "run");
                runStart = i;
            }
        }
    }

    if (fieldName == "size" && isCyclicStrip) {
        const int topY = 0;
        const int roiH = std::clamp(static_cast<int>(extended.rows * 0.34), 48, std::max(48, extended.rows / 2));
        const int roiW = std::clamp(static_cast<int>(stripImage.cols * 0.16), 260, std::max(260, stripImage.cols / 4));
        const int margin = std::clamp(static_cast<int>(stripImage.cols * 0.03), 24, 160);
        const std::vector<int> xPositions = {
            overlap + margin,
            overlap + margin + roiW / 2,
            overlap + stripImage.cols - margin - roiW,
            overlap + stripImage.cols - margin - roiW - roiW / 2
        };

        for (std::size_t idx = 0; idx < xPositions.size(); ++idx) {
            cv::Rect rect(xPositions[idx], topY, roiW, roiH);
            rect = clampRect(rect, extended.size());
            if (rect.width < 120 || rect.height < 40) {
                continue;
            }
            const cv::Rect mappedRect((rect.x - overlap + stripImage.cols) % stripImage.cols,
                                      rect.y,
                                      std::min(rect.width, stripImage.cols),
                                      rect.height);
            const cv::Mat crop = extended(rect).clone();
            const double roiQuality = preprocessor_.computeImageQualityScore(preprocessor_.toGrayscale(crop));

            StripCandidate candidate;
            candidate.name = "size_edge_" + std::to_string(idx);
            candidate.extendedBox = rect;
            candidate.mappedBox = mappedRect;
            candidate.image = crop;
            candidate.geometryScore = 0.62;
            candidate.imageQualityScore = roiQuality;
            candidates.push_back(std::move(candidate));
        }
    }

    if (candidates.empty()) {
        cv::Mat columnEnergy;
        cv::reduce(absGradX, columnEnergy, 0, cv::REDUCE_SUM, CV_32F);
        std::vector<float> energy(columnEnergy.cols, 0.0F);
        for (int x = 0; x < columnEnergy.cols; ++x) {
            energy[static_cast<std::size_t>(x)] = columnEnergy.at<float>(0, x);
        }

        std::vector<int> centers;
        const int desired = fieldName == "size" ? 4 : 6;
        const int windowWidth = fieldName == "size"
            ? std::clamp(extended.cols / 32, 160, 540)
            : std::clamp(extended.cols / 48, 100, 280);
        const int suppressionRadius = std::max(60, windowWidth / 2);

        std::vector<float> energyWork = energy;
        for (int k = 0; k < desired; ++k) {
            auto bestIt = std::max_element(energyWork.begin(), energyWork.end());
            if (bestIt == energyWork.end() || *bestIt <= 0.0F) {
                break;
            }
            const int centerX = static_cast<int>(std::distance(energyWork.begin(), bestIt));
            centers.push_back(centerX);
            const int x0 = std::max(0, centerX - suppressionRadius);
            const int x1 = std::min(static_cast<int>(energyWork.size()), centerX + suppressionRadius);
            for (int x = x0; x < x1; ++x) {
                energyWork[static_cast<std::size_t>(x)] = 0.0F;
            }
        }

        if (centers.empty()) {
            centers = {
                extended.cols / 8,
                extended.cols / 3,
                extended.cols / 2,
                (extended.cols * 2) / 3,
                (extended.cols * 7) / 8
            };
        }

        for (std::size_t idx = 0; idx < centers.size(); ++idx) {
            cv::Rect rect(centers[idx] - windowWidth / 2,
                          std::max(0, extended.rows / 8),
                          windowWidth,
                          std::max(32, (extended.rows * 6) / 8));
            rect = clampRect(rect, extended.size());
            if (rect.width < 60 || rect.height < 24) {
                continue;
            }

            const cv::Rect mappedRect = isCyclicStrip
                ? cv::Rect((rect.x - overlap + stripImage.cols) % stripImage.cols,
                           rect.y,
                           std::min(rect.width, stripImage.cols),
                           rect.height)
                : rect;

            const cv::Mat crop = extended(rect).clone();
            const double roiQuality = preprocessor_.computeImageQualityScore(preprocessor_.toGrayscale(crop));

            StripCandidate candidate;
            candidate.name = fieldName + "_energy_" + std::to_string(idx);
            candidate.extendedBox = rect;
            candidate.mappedBox = mappedRect;
            candidate.image = crop;
            candidate.geometryScore = 0.30;
            candidate.imageQualityScore = roiQuality;
            candidates.push_back(std::move(candidate));
        }
    }

    std::sort(candidates.begin(), candidates.end(), [](const StripCandidate& a, const StripCandidate& b) {
        return (a.geometryScore + a.imageQualityScore) > (b.geometryScore + b.imageQualityScore);
    });

    const std::size_t limit = fieldName == "size" ? 6U : 8U;
    if (candidates.size() > limit) {
        candidates.resize(limit);
    }

    if (saveDebugArtifacts_) {
        for (std::size_t i = 0; i < candidates.size(); ++i) {
            cv::rectangle(overlay, candidates[i].extendedBox, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            cv::putText(overlay,
                        std::to_string(i),
                        candidates[i].extendedBox.tl() + cv::Point(0, -4),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.6,
                        cv::Scalar(0, 255, 255),
                        2,
                        cv::LINE_AA);
            saveDebugImage(candidates[i].image,
                           (fs::path(debugDir) / ("55_" + fieldName + "_candidate_" + std::to_string(i) + ".png")).string());
        }
        saveDebugImage(overlay, (fs::path(debugDir) / ("56_" + fieldName + "_candidate_overlay.png")).string());

        if (fieldName == "size") {
            const fs::path sizeDumpDir = fs::path(debugDir) / "size_candidates";
            fs::create_directories(sizeDumpDir);

            std::vector<StripCandidate> ranked = candidates;
            std::sort(ranked.begin(), ranked.end(), [](const StripCandidate& a, const StripCandidate& b) {
                return a.geometryScore > b.geometryScore;
            });
            if (ranked.size() > 10) {
                ranked.resize(10);
            }

            std::ofstream meta(sizeDumpDir / "summary.csv");
            meta << "index,name,x,y,width,height,structure_score,roi_quality,selection_reason\n";
            for (std::size_t i = 0; i < ranked.size(); ++i) {
                saveDebugImage(ranked[i].image, (sizeDumpDir / ("roi_" + std::to_string(i) + ".png")).string());
                meta << i << ","
                     << sanitizeCsvField(ranked[i].name) << ","
                     << ranked[i].mappedBox.x << ","
                     << ranked[i].mappedBox.y << ","
                     << ranked[i].mappedBox.width << ","
                     << ranked[i].mappedBox.height << ","
                     << formatDouble(ranked[i].geometryScore) << ","
                     << formatDouble(ranked[i].imageQualityScore) << ","
                     << sanitizeCsvField("top_size_candidate") << "\n";
            }

            cv::Mat topOverlay;
            cv::cvtColor(gray, topOverlay, cv::COLOR_GRAY2BGR);
            for (std::size_t i = 0; i < ranked.size(); ++i) {
                cv::rectangle(topOverlay, ranked[i].extendedBox, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
                cv::putText(topOverlay,
                            std::to_string(i) + " s=" + formatDouble(ranked[i].geometryScore, 2),
                            ranked[i].extendedBox.tl() + cv::Point(0, -6),
                            cv::FONT_HERSHEY_SIMPLEX,
                            0.55,
                            cv::Scalar(0, 255, 0),
                            2,
                            cv::LINE_AA);
            }
            saveDebugImage(topOverlay, (sizeDumpDir / "top10_overlay.png").string());
        }
    }

    return candidates;
}

FieldResult TyreAnalyzer::detectTyreSizeField(const cv::Mat& image,
                                              const std::string& debugDir,
                                              AnalysisResult& result) const {
    FieldResult best;
    if (!ocrEngine_.isInitialized() || image.empty()) {
        return best;
    }

    const std::string whitelist = "0123456789RZVWHY/ -";
    std::vector<OcrProbe> probes;
    if (image.cols > image.rows * 4) {
        const auto stripCandidates = detectSidewallCandidates(image, "size", debugDir, result);
        for (const auto& candidate : stripCandidates) {
            probes.push_back({candidate.name, candidate.mappedBox, candidate.image});
        }
    } else {
        probes = buildStripProbes(image, "size");
    }
    if (probes.size() > 5) {
        probes.resize(5);
    }
    double buildMs = 0.0;
    double ocrMs = 0.0;

    std::ofstream report;
    if (saveDebugArtifacts_) {
        report.open(result.ocrReportPath, std::ios::app);
    }

    auto logRow = [&](const std::string& roiId,
                      const std::string& variant,
                      const std::string& psm,
                      const cv::Size& size,
                      double elapsed,
                      double ocrConf,
                      bool regexValid,
                      double domainScore,
                      double finalScore,
                      const std::string& status,
                      const std::string& text,
                      const std::string& normalized) {
        if (!report.is_open()) {
            return;
        }
        report << "size," << sanitizeCsvField(roiId) << ","
               << sanitizeCsvField(variant) << ","
               << sanitizeCsvField(psm) << ","
               << size.width << ","
               << size.height << ","
               << formatDouble(elapsed) << ","
               << formatDouble(ocrConf) << ","
               << (regexValid ? "true" : "false") << ","
               << formatDouble(domainScore) << ","
               << formatDouble(finalScore) << ","
               << sanitizeCsvField(status) << ","
               << sanitizeCsvField(text) << ","
               << sanitizeCsvField(normalized) << "\n";
    };

    for (std::size_t probeIndex = 0; probeIndex < probes.size(); ++probeIndex) {
        if (saveDebugArtifacts_) {
            saveDebugImage(probes[probeIndex].image,
                           (fs::path(debugDir) / ("30_size_probe_" + std::to_string(probeIndex) + ".png")).string());
        }

        const Clock::time_point buildStart = Clock::now();
        const auto variants = buildFastVariants(probes[probeIndex].image, false);
        buildMs += elapsedMs(buildStart, Clock::now());
        std::vector<std::pair<std::string, cv::Mat>> limitedVariants = variants;
        if (limitedVariants.size() > 3) {
            limitedVariants.resize(3);
        }
        if (saveDebugArtifacts_) {
            for (const auto& variant : limitedVariants) {
                saveDebugImage(variant.second,
                               (fs::path(debugDir) / ("31_size_" + std::to_string(probeIndex) + "_" + variant.first + ".png")).string());
            }
        }

        for (const auto& variant : limitedVariants) {
            const Clock::time_point ocrStart = Clock::now();
            const OcrResult ocr = ocrEngine_.recognize(variant.second, variant.first, tesseract::PSM_SINGLE_LINE, whitelist);
            const double callMs = elapsedMs(ocrStart, Clock::now());
            ocrMs += callMs;
            const ParsedSize parsed = parseTyreSize(ocr.text);
            const double roiQuality = preprocessor_.computeImageQualityScore(preprocessor_.toGrayscale(probes[probeIndex].image));
            const double domainScore = parsed.parseQuality;
            if (!parsed.found) {
                logRow(probes[probeIndex].name, variant.first, "single_line", variant.second.size(), callMs,
                       ocr.averageConfidence, false, domainScore, 0.0, "parse_miss", ocr.text, "");
                continue;
            }
            const double confidence = clamp01(0.50 * ocr.averageConfidence + 0.35 * parsed.parseQuality + 0.15 * roiQuality);
            const std::string status = confidence > best.confidence ? "candidate_best" : "candidate_weaker";
            logRow(probes[probeIndex].name, variant.first, "single_line", variant.second.size(), callMs,
                   ocr.averageConfidence, true, domainScore, confidence, status, ocr.text, parsed.normalized);
            if (confidence > best.confidence) {
                best.rawText = ocr.text;
                best.normalizedText = parsed.normalized;
                best.found = true;
                best.confidence = confidence;
                best.uncertainty = 1.0 - confidence;
                best.roiQuality = roiQuality;
                best.boundingBox = probes[probeIndex].roi;
            }

            if (best.confidence >= 0.84) {
                appendTiming(result.stepTimings, "size_variants_build_ms", buildMs);
                appendTiming(result.stepTimings, "size_tesseract_ms", ocrMs);
                result.notes.push_back("Size branch early-stopped on high-score candidate.");
                return best;
            }
        }
    }

    appendTiming(result.stepTimings, "size_variants_build_ms", buildMs);
    appendTiming(result.stepTimings, "size_tesseract_ms", ocrMs);
    return best;
}

FieldResult TyreAnalyzer::detectDotField(const cv::Mat& image,
                                         const std::string& debugDir,
                                         AnalysisResult& result) const {
    FieldResult best;
    if (!ocrEngine_.isInitialized() || image.empty()) {
        return best;
    }

    const std::string whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -";
    std::vector<OcrProbe> probes;
    if (image.cols > image.rows * 4) {
        const auto stripCandidates = detectSidewallCandidates(image, "dot", debugDir, result);
        for (const auto& candidate : stripCandidates) {
            probes.push_back({candidate.name, candidate.mappedBox, candidate.image});
        }
    } else {
        probes = buildStripProbes(image, "dot");
    }
    if (probes.size() > 6) {
        probes.resize(6);
    }
    double buildMs = 0.0;
    double ocrMs = 0.0;

    std::ofstream report;
    if (saveDebugArtifacts_) {
        report.open(result.ocrReportPath, std::ios::app);
    }

    auto logRow = [&](const std::string& roiId,
                      const std::string& variant,
                      const std::string& psm,
                      const cv::Size& size,
                      double elapsed,
                      double ocrConf,
                      bool regexValid,
                      double domainScore,
                      double finalScore,
                      const std::string& status,
                      const std::string& text,
                      const std::string& normalized) {
        if (!report.is_open()) {
            return;
        }
        report << "dot," << sanitizeCsvField(roiId) << ","
               << sanitizeCsvField(variant) << ","
               << sanitizeCsvField(psm) << ","
               << size.width << ","
               << size.height << ","
               << formatDouble(elapsed) << ","
               << formatDouble(ocrConf) << ","
               << (regexValid ? "true" : "false") << ","
               << formatDouble(domainScore) << ","
               << formatDouble(finalScore) << ","
               << sanitizeCsvField(status) << ","
               << sanitizeCsvField(text) << ","
               << sanitizeCsvField(normalized) << "\n";
    };

    for (std::size_t probeIndex = 0; probeIndex < probes.size(); ++probeIndex) {
        if (saveDebugArtifacts_) {
            saveDebugImage(probes[probeIndex].image,
                           (fs::path(debugDir) / ("40_dot_probe_" + std::to_string(probeIndex) + ".png")).string());
        }

        const Clock::time_point buildStart = Clock::now();
        const auto variants = buildFastVariants(probes[probeIndex].image, true);
        buildMs += elapsedMs(buildStart, Clock::now());
        std::vector<std::pair<std::string, cv::Mat>> limitedVariants = variants;
        if (limitedVariants.size() > 3) {
            limitedVariants.resize(3);
        }
        if (saveDebugArtifacts_) {
            for (const auto& variant : limitedVariants) {
                saveDebugImage(variant.second,
                               (fs::path(debugDir) / ("41_dot_" + std::to_string(probeIndex) + "_" + variant.first + ".png")).string());
            }
        }

        for (const auto& variant : limitedVariants) {
            const Clock::time_point blockStart = Clock::now();
            const OcrResult ocr = ocrEngine_.recognize(variant.second, variant.first, tesseract::PSM_SINGLE_LINE, whitelist);
            const double callMs = elapsedMs(blockStart, Clock::now());
            ocrMs += callMs;
            const ParsedDot parsed = parseDot(ocr.text);
            const double roiQuality = preprocessor_.computeImageQualityScore(preprocessor_.toGrayscale(probes[probeIndex].image));
            const double completeness = parsed.fullFound ? 1.0 : (parsed.weekYearFound ? 0.78 : (parsed.dotFound ? 0.55 : 0.0));
            if (!parsed.dotFound && parsed.weekYear.empty()) {
                logRow(probes[probeIndex].name, variant.first, "single_line", variant.second.size(), callMs,
                       ocr.averageConfidence, false, completeness, 0.0, "parse_miss", ocr.text, "");
                continue;
            }
            const double confidence = clamp01(0.45 * ocr.averageConfidence + 0.35 * completeness + 0.20 * roiQuality);
            const std::string normalized = parsed.normalized.empty() ? parsed.fullNormalized : parsed.normalized;
            const std::string status = confidence > best.confidence ? "candidate_best" : "candidate_weaker";
            logRow(probes[probeIndex].name, variant.first, "single_line", variant.second.size(), callMs,
                   ocr.averageConfidence, true, completeness, confidence, status, ocr.text, normalized);
            if (confidence > best.confidence) {
                best.rawText = parsed.raw.empty() ? ocr.text : parsed.raw;
                best.normalizedText = normalized;
                best.found = parsed.dotFound || parsed.weekYearFound;
                best.confidence = confidence;
                best.uncertainty = 1.0 - confidence;
                best.roiQuality = roiQuality;
                best.boundingBox = probes[probeIndex].roi;
            }

            if (parsed.weekYearFound && confidence >= 0.76) {
                appendTiming(result.stepTimings, "dot_variants_build_ms", buildMs);
                appendTiming(result.stepTimings, "dot_tesseract_ms", ocrMs);
                result.notes.push_back("DOT branch early-stopped on valid week/year candidate.");
                return best;
            }
        }
    }

    appendTiming(result.stepTimings, "dot_variants_build_ms", buildMs);
    appendTiming(result.stepTimings, "dot_tesseract_ms", ocrMs);
    return best;
}

TyreAnalyzer::ParsedSize TyreAnalyzer::parseTyreSize(const std::string& text) const {
    ParsedSize result;
    const std::string sanitized = sanitizeForParsing(text);
    const std::regex sizeRegex(R"((^|[^0-9])([1-3][0-9]{2})\s*[/\\]?\s*([2-9][0-9])\s*([RZ]?)\s*([1-2][0-9])(?:\s*([A-Z]))?([^0-9A-Z]|$))");
    std::smatch match;
    if (std::regex_search(sanitized, match, sizeRegex)) {
        const std::string width = match[2].str();
        const std::string aspect = match[3].str();
        const std::string construction = match[4].str().empty() ? "R" : match[4].str();
        const std::string rim = match[5].str();
        const std::string speed = match[6].str();

        const int widthValue = std::stoi(width);
        const int aspectValue = std::stoi(aspect);
        const int rimValue = std::stoi(rim);

        if (widthValue >= 125 && widthValue <= 395 &&
            aspectValue >= 20 && aspectValue <= 95 &&
            rimValue >= 10 && rimValue <= 24) {
            result.found = true;
            result.raw = squeezeSpaces(text);
            result.normalized = width + "/" + aspect + " " + construction + rim;
            if (!speed.empty()) {
                result.normalized += " " + speed;
            }
            result.parseQuality = match[0].str().find('/') != std::string::npos ? 1.0 : 0.88;
            if (!speed.empty()) {
                result.parseQuality = std::min(1.0, result.parseQuality + 0.03);
            }
        }
    }
    return result;
}

TyreAnalyzer::ParsedDot TyreAnalyzer::parseDot(const std::string& text) const {
    ParsedDot result;
    const std::string compactOriginal = normalizeDotToken(text);
    const std::string compactDigits = normalizeDotToken(sanitizeForParsing(text));

    const std::regex dotRegex(R"(DOT\s*([A-Z0-9 ]{4,24}))");
    std::smatch dotMatch;
    if (std::regex_search(compactOriginal, dotMatch, dotRegex)) {
        const std::string dotCompact = normalizeDotToken(dotMatch[1].str());
        result.dotFound = true;
        result.raw = squeezeSpaces(text);
        result.fullRaw = "DOT " + squeezeSpaces(dotMatch[1].str());
        result.fullNormalized = "DOT" + dotCompact;
        result.normalized = result.fullNormalized;
        result.fullFound = dotCompact.size() >= 8;
    } else {
        result.raw = squeezeSpaces(text);
    }

    const std::regex weekYearRegex(R"((\d{4})(?!.*\d))");
    std::smatch weekYearMatch;
    if (std::regex_search(compactDigits, weekYearMatch, weekYearRegex) && isLikelyDotWeekYear(weekYearMatch[1].str())) {
        result.weekYearFound = true;
        result.dotFound = true;
        result.weekYear = weekYearMatch[1].str();
        if (result.normalized.empty()) {
            result.normalized = result.weekYear;
        }
    }

    if (result.fullFound && result.weekYearFound) {
        result.parseQuality = 1.0;
    } else if (result.weekYearFound && result.dotFound) {
        result.parseQuality = 0.82;
    } else if (result.weekYearFound) {
        result.parseQuality = 0.68;
    } else if (result.dotFound) {
        result.parseQuality = 0.55;
    }

    return result;
}

double TyreAnalyzer::computeSizeConfidence(const ParsedSize& parsed, const CandidateText& candidate) const {
    const double score = 0.45 * candidate.ocr.averageConfidence +
                         0.30 * parsed.parseQuality +
                         0.15 * candidate.roi.imageQualityScore +
                         0.10 * candidate.roi.geometryScore;
    return clamp01(score);
}

double TyreAnalyzer::computeDotConfidence(const ParsedDot& parsed, const CandidateText& candidate) const {
    double completeness = parsed.parseQuality;
    if (parsed.weekYearFound && !parsed.fullFound) {
        completeness = std::max(completeness, 0.72);
    }
    const double score = 0.42 * candidate.ocr.averageConfidence +
                         0.33 * completeness +
                         0.15 * candidate.roi.imageQualityScore +
                         0.10 * candidate.roi.geometryScore;
    return clamp01(score);
}

std::vector<TyreAnalyzer::CandidateText> TyreAnalyzer::collectCandidateTexts(
    const cv::Mat& frame,
    const std::vector<CandidateRoi>& rois,
    const std::string& debugDir,
    AnalysisResult& result,
    std::vector<cv::Rect>& overlayBoxes) const {
    std::vector<CandidateText> candidates;

    if (!ocrEngine_.isInitialized()) {
        result.notes.push_back("Tesseract OCR initialization failed. OCR results are unavailable.");
        return candidates;
    }

    std::ofstream ocrReport;
    if (saveDebugArtifacts_) {
        fs::create_directories(debugDir);
        ocrReport.open(result.ocrReportPath);
        ocrReport << "roi_index,variant,psm,confidence,text,size_found,dot_found,week_year,full_dot\n";
    }

    double variantsBuildMs = 0.0;
    double tesseractMs = 0.0;
    for (const auto& roi : rois) {
        const std::size_t roiIndex = static_cast<std::size_t>(&roi - rois.data());
        const cv::Rect bounded = roi.box & cv::Rect(0, 0, frame.cols, frame.rows);
        if (bounded.empty()) {
            continue;
        }
        if (bounded.width < 48 || bounded.height < 16) {
            continue;
        }

        const double aspectRatio = static_cast<double>(bounded.width) / std::max(1, bounded.height);
        if (aspectRatio < 1.2 || aspectRatio > 80.0) {
            continue;
        }

        overlayBoxes.push_back(bounded);
        const cv::Mat crop = frame(bounded).clone();
        if (saveDebugArtifacts_) {
            saveDebugImage(crop, (fs::path(debugDir) / ("20_roi_" + std::to_string(roiIndex) + "_crop.png")).string());
        }
        const Clock::time_point variantsStart = Clock::now();
        const auto variants = preprocessor_.buildOcrVariants(crop);
        variantsBuildMs += elapsedMs(variantsStart, Clock::now());
        if (variants.empty()) {
            continue;
        }
        if (saveDebugArtifacts_) {
            for (const auto& variant : variants) {
                saveDebugImage(variant.second,
                               (fs::path(debugDir) / ("21_roi_" + std::to_string(roiIndex) + "_" + variant.first + ".png")).string());
            }
        }

        const Clock::time_point blockOcrStart = Clock::now();
        std::vector<OcrResult> blockResults = ocrEngine_.recognizeVariants(variants, tesseract::PSM_SINGLE_BLOCK);
        for (auto& resultItem : blockResults) {
            resultItem.variantName = "block:" + resultItem.variantName;
        }
        tesseractMs += elapsedMs(blockOcrStart, Clock::now());
        if (aspectRatio >= 4.0 && roi.imageQualityScore >= 0.12) {
            const Clock::time_point lineOcrStart = Clock::now();
            std::vector<OcrResult> lineResults = ocrEngine_.recognizeVariants(variants, tesseract::PSM_SINGLE_LINE);
            for (auto& resultItem : lineResults) {
                resultItem.variantName = "line:" + resultItem.variantName;
            }
            tesseractMs += elapsedMs(lineOcrStart, Clock::now());
            blockResults.insert(blockResults.end(), lineResults.begin(), lineResults.end());
        }

        for (const auto& ocrResult : blockResults) {
            if (ocrResult.text.empty()) {
                continue;
            }
            CandidateText candidate;
            candidate.roi = roi;
            candidate.ocr = ocrResult;
            candidate.size = parseTyreSize(ocrResult.text);
            candidate.dot = parseDot(ocrResult.text);
            if (ocrReport.is_open()) {
                const bool isLine = ocrResult.variantName.rfind("line:", 0) == 0;
                ocrReport << roiIndex << ","
                          << sanitizeCsvField(ocrResult.variantName) << ","
                          << sanitizeCsvField(isLine ? "single_line" : "single_block") << ","
                          << formatDouble(ocrResult.averageConfidence) << ","
                          << sanitizeCsvField(ocrResult.text) << ","
                          << (candidate.size.found ? "true" : "false") << ","
                          << (candidate.dot.dotFound ? "true" : "false") << ","
                          << sanitizeCsvField(candidate.dot.weekYear) << ","
                          << sanitizeCsvField(candidate.dot.fullNormalized) << "\n";
            }
            candidates.push_back(std::move(candidate));
        }
    }

    appendTiming(result.stepTimings, "ocr_variants_build_ms", variantsBuildMs);
    appendTiming(result.stepTimings, "ocr_tesseract_ms", tesseractMs);

    if (candidates.empty()) {
        result.notes.push_back("OCR did not produce usable text on any ROI.");
    }
    return candidates;
}

void TyreAnalyzer::saveCropOrPlaceholder(const cv::Mat& source,
                                         const cv::Rect& roi,
                                         const std::string& path,
                                         const std::string& label,
                                         bool roiFound) const {
    if (roiFound && !roi.empty()) {
        const cv::Rect bounded = roi & cv::Rect(0, 0, source.cols, source.rows);
        if (!bounded.empty()) {
            cv::imwrite(path, source(bounded));
            return;
        }
    }

    cv::Mat placeholder(140, 520, CV_8UC3, cv::Scalar(20, 20, 20));
    cv::putText(placeholder, label, cv::Point(16, 72), cv::FONT_HERSHEY_SIMPLEX, 0.75,
                cv::Scalar(220, 220, 220), 2, cv::LINE_AA);
    cv::imwrite(path, placeholder);
}

void TyreAnalyzer::saveOverlay(const cv::Mat& image,
                               const std::vector<cv::Rect>& roiBoxes,
                               const cv::Rect& sizeBox,
                               const cv::Rect& dotBox,
                               const std::string& path) const {
    cv::Mat overlay = image.clone();
    for (const auto& box : roiBoxes) {
        cv::rectangle(overlay, box, cv::Scalar(255, 200, 0), 2);
    }
    if (!sizeBox.empty()) {
        cv::rectangle(overlay, sizeBox, cv::Scalar(0, 255, 0), 3);
        cv::putText(overlay, "SIZE", sizeBox.tl() + cv::Point(0, -6), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
    if (!dotBox.empty()) {
        cv::rectangle(overlay, dotBox, cv::Scalar(0, 140, 255), 3);
        cv::putText(overlay, "DOT", dotBox.tl() + cv::Point(0, -6), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 140, 255), 2, cv::LINE_AA);
    }
    cv::imwrite(path, overlay);
}

}  // namespace tyre
