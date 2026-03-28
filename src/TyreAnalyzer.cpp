#include "TyreAnalyzer.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace fs = std::filesystem;

namespace tyre {

namespace {

using Clock = std::chrono::steady_clock;

double elapsedMs(const Clock::time_point& start, const Clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

bool isLikelyDotWeekYear(const std::string& weekYear) {
    if (weekYear.size() != 4) {
        return false;
    }
    const int week = std::stoi(weekYear.substr(0, 2));
    return week >= 1 && week <= 53;
}

}  // namespace

TyreAnalyzer::TyreAnalyzer(bool saveDebugArtifacts)
    : ocrEngine_(TesseractOcrEngine::Settings{}),
      saveDebugArtifacts_(saveDebugArtifacts) {
}

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
    if (wheelGeometry.found) {
        const Clock::time_point unwrapStart = Clock::now();
        cv::Mat sidewallBand =
            preprocessor_.unwrapSidewallBand(frame, wheelGeometry, saveDebugArtifacts_ ? &wheelDebugImages : nullptr, &result.stepTimings);
        appendTiming(result.stepTimings, "wheel_unwrap_total_ms", elapsedMs(unwrapStart, Clock::now()));
        if (!sidewallBand.empty()) {
            ocrSource = sidewallBand;
            result.notes.push_back("Using unwrapped sidewall band for OCR.");
        } else {
            result.notes.push_back("Wheel detected, but sidewall unwrap failed. Falling back to full image.");
        }
        if (saveDebugArtifacts_) {
            saveDebugImage(wheelDebugImages.polarFull, (fs::path(debugDir) / "09_polar_full.png").string());
            saveDebugImage(wheelDebugImages.sidewallBand, (fs::path(debugDir) / "10_sidewall_band.png").string());
        }
    } else {
        result.notes.push_back("Wheel circle not detected. Falling back to full image.");
    }

    if (saveDebugArtifacts_) {
        saveDebugImage(ocrSource, (fs::path(debugDir) / "11_ocr_source.png").string());
    }

    std::vector<cv::Rect> overlayBoxes;
    const Clock::time_point sizeStart = Clock::now();
    result.tyreSize = detectTyreSizeField(ocrSource, debugDir, result);
    result.timings.roiProposalMs = elapsedMs(sizeStart, Clock::now());
    if (result.tyreSize.found) {
        result.tyreSizeFound = true;
        overlayBoxes.push_back(result.tyreSize.boundingBox);
    }

    const Clock::time_point dotStart = Clock::now();
    result.dot = detectDotField(ocrSource, debugDir, result);
    result.timings.ocrMs = elapsedMs(dotStart, Clock::now());
    if (result.dot.found) {
        result.dotFound = true;
        overlayBoxes.push_back(result.dot.boundingBox);
        const ParsedDot parsedDot = parseDot(result.dot.rawText);
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
    saveCropOrPlaceholder(ocrSource, result.tyreSize.boundingBox, sizeCropPath, "SIZE ROI NOT FOUND", result.tyreSizeFound);
    saveCropOrPlaceholder(ocrSource, result.dot.boundingBox, dotCropPath, "DOT ROI NOT FOUND", result.dotFound);
    result.tyreSize.cropPath = sizeCropPath;
    result.dot.cropPath = dotCropPath;
    result.timings.cropSaveMs = elapsedMs(cropStart, Clock::now());

    const Clock::time_point overlayStart = Clock::now();
    saveOverlay(ocrSource, overlayBoxes, result.tyreSize.boundingBox, result.dot.boundingBox, overlayPath);
    result.overlayPath = overlayPath;
    result.timings.overlaySaveMs = elapsedMs(overlayStart, Clock::now());

    result.timings.totalMs = elapsedMs(totalStart, Clock::now());
    appendTiming(result.stepTimings, "total_ms", result.timings.totalMs);
    if (saveDebugArtifacts_) {
        cv::Mat roiOverlay = ocrSource.clone();
        for (std::size_t index = 0; index < overlayBoxes.size(); ++index) {
            cv::rectangle(roiOverlay, overlayBoxes[index], cv::Scalar(255, 200, 0), 2);
            cv::putText(roiOverlay, std::to_string(index), overlayBoxes[index].tl() + cv::Point(0, -4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 200, 0), 2, cv::LINE_AA);
        }
        saveDebugImage(roiOverlay, (fs::path(debugDir) / "20_roi_overlay.png").string());
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
    if (gray.rows < 72) {
        gray = preprocessor_.resizeUpscale(gray, 2.0);
    }
    const cv::Mat clahe = preprocessor_.applyClahe(gray);
    const cv::Mat adaptive = preprocessor_.adaptiveThresholdImage(clahe);
    variants.emplace_back("gray", gray);
    variants.emplace_back("clahe", clahe);
    variants.emplace_back("adaptive", adaptive);
    if (aggressiveThreshold) {
        variants.emplace_back("adaptive_inv", preprocessor_.invertImage(adaptive));
    }
    return variants;
}

FieldResult TyreAnalyzer::detectTyreSizeField(const cv::Mat& image,
                                              const std::string& debugDir,
                                              AnalysisResult& result) const {
    FieldResult best;
    if (!ocrEngine_.isInitialized() || image.empty()) {
        return best;
    }

    const std::string whitelist = "0123456789RZVWHY/ -";
    const auto probes = buildStripProbes(image, "size");
    double buildMs = 0.0;
    double ocrMs = 0.0;

    std::ofstream report;
    if (saveDebugArtifacts_) {
        report.open(result.ocrReportPath, std::ios::app);
        if (report.tellp() == 0) {
            report << "field,probe,variant,psm,confidence,text,normalized\n";
        }
    }

    for (std::size_t probeIndex = 0; probeIndex < probes.size(); ++probeIndex) {
        if (saveDebugArtifacts_) {
            saveDebugImage(probes[probeIndex].image,
                           (fs::path(debugDir) / ("30_size_probe_" + std::to_string(probeIndex) + ".png")).string());
        }

        const Clock::time_point buildStart = Clock::now();
        const auto variants = buildFastVariants(probes[probeIndex].image, false);
        buildMs += elapsedMs(buildStart, Clock::now());
        if (saveDebugArtifacts_) {
            for (const auto& variant : variants) {
                saveDebugImage(variant.second,
                               (fs::path(debugDir) / ("31_size_" + std::to_string(probeIndex) + "_" + variant.first + ".png")).string());
            }
        }

        const Clock::time_point ocrStart = Clock::now();
        auto lineResults = ocrEngine_.recognizeVariants(variants, tesseract::PSM_SINGLE_LINE, whitelist);
        auto wordResults = ocrEngine_.recognizeVariants(variants, tesseract::PSM_SINGLE_WORD, whitelist);
        ocrMs += elapsedMs(ocrStart, Clock::now());
        lineResults.insert(lineResults.end(), wordResults.begin(), wordResults.end());

        for (const auto& ocr : lineResults) {
            const ParsedSize parsed = parseTyreSize(ocr.text);
            if (!parsed.found) {
                continue;
            }
            const double roiQuality = preprocessor_.computeImageQualityScore(preprocessor_.toGrayscale(probes[probeIndex].image));
            const double confidence = clamp01(0.50 * ocr.averageConfidence + 0.35 * parsed.parseQuality + 0.15 * roiQuality);
            if (report.is_open()) {
                report << "size," << sanitizeCsvField(probes[probeIndex].name) << ","
                       << sanitizeCsvField(ocr.variantName) << ",\"single_line\","
                       << formatDouble(ocr.averageConfidence) << ","
                       << sanitizeCsvField(ocr.text) << ","
                       << sanitizeCsvField(parsed.normalized) << "\n";
            }
            if (confidence > best.confidence) {
                best.rawText = ocr.text;
                best.normalizedText = parsed.normalized;
                best.found = true;
                best.confidence = confidence;
                best.uncertainty = 1.0 - confidence;
                best.roiQuality = roiQuality;
                best.boundingBox = probes[probeIndex].roi;
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
    const auto probes = buildStripProbes(image, "dot");
    double buildMs = 0.0;
    double ocrMs = 0.0;

    std::ofstream report;
    if (saveDebugArtifacts_) {
        report.open(result.ocrReportPath, std::ios::app);
        if (report.tellp() == 0) {
            report << "field,probe,variant,psm,confidence,text,normalized\n";
        }
    }

    for (std::size_t probeIndex = 0; probeIndex < probes.size(); ++probeIndex) {
        if (saveDebugArtifacts_) {
            saveDebugImage(probes[probeIndex].image,
                           (fs::path(debugDir) / ("40_dot_probe_" + std::to_string(probeIndex) + ".png")).string());
        }

        const Clock::time_point buildStart = Clock::now();
        const auto variants = buildFastVariants(probes[probeIndex].image, true);
        buildMs += elapsedMs(buildStart, Clock::now());
        if (saveDebugArtifacts_) {
            for (const auto& variant : variants) {
                saveDebugImage(variant.second,
                               (fs::path(debugDir) / ("41_dot_" + std::to_string(probeIndex) + "_" + variant.first + ".png")).string());
            }
        }

        const Clock::time_point blockStart = Clock::now();
        auto blockResults = ocrEngine_.recognizeVariants(variants, tesseract::PSM_SINGLE_LINE, whitelist);
        auto wordResults = ocrEngine_.recognizeVariants(variants, tesseract::PSM_SINGLE_WORD, whitelist);
        ocrMs += elapsedMs(blockStart, Clock::now());
        blockResults.insert(blockResults.end(), wordResults.begin(), wordResults.end());

        for (const auto& ocr : blockResults) {
            const ParsedDot parsed = parseDot(ocr.text);
            if (!parsed.dotFound && parsed.weekYear.empty()) {
                continue;
            }
            const double roiQuality = preprocessor_.computeImageQualityScore(preprocessor_.toGrayscale(probes[probeIndex].image));
            const double completeness = parsed.fullFound ? 1.0 : (parsed.weekYearFound ? 0.78 : 0.55);
            const double confidence = clamp01(0.45 * ocr.averageConfidence + 0.35 * completeness + 0.20 * roiQuality);
            if (report.is_open()) {
                report << "dot," << sanitizeCsvField(probes[probeIndex].name) << ","
                       << sanitizeCsvField(ocr.variantName) << ","
                       << sanitizeCsvField("mixed") << ","
                       << formatDouble(ocr.averageConfidence) << ","
                       << sanitizeCsvField(ocr.text) << ","
                       << sanitizeCsvField(parsed.normalized.empty() ? parsed.fullNormalized : parsed.normalized) << "\n";
            }
            if (confidence > best.confidence) {
                best.rawText = parsed.raw.empty() ? ocr.text : parsed.raw;
                best.normalizedText = parsed.normalized.empty() ? parsed.fullNormalized : parsed.normalized;
                best.found = parsed.dotFound || parsed.weekYearFound;
                best.confidence = confidence;
                best.uncertainty = 1.0 - confidence;
                best.roiQuality = roiQuality;
                best.boundingBox = probes[probeIndex].roi;
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
    const std::regex sizeRegex(R"((^|[^0-9])([1-3][0-9]{2})\s*[/\\]?\s*([2-9][0-9])\s*([RZ]?)\s*([1-2][0-9])([^0-9]|$))");
    std::smatch match;
    if (std::regex_search(sanitized, match, sizeRegex)) {
        const std::string width = match[2].str();
        const std::string aspect = match[3].str();
        const std::string construction = match[4].str().empty() ? "R" : match[4].str();
        const std::string rim = match[5].str();

        const int widthValue = std::stoi(width);
        const int aspectValue = std::stoi(aspect);
        const int rimValue = std::stoi(rim);

        if (widthValue >= 125 && widthValue <= 395 &&
            aspectValue >= 20 && aspectValue <= 95 &&
            rimValue >= 10 && rimValue <= 24) {
            result.found = true;
            result.raw = squeezeSpaces(text);
            result.normalized = width + "/" + aspect + " " + construction + rim;
            result.parseQuality = match[0].str().find('/') != std::string::npos ? 1.0 : 0.88;
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
