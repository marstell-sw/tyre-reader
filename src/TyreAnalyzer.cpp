#include "TyreAnalyzer.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
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

TyreAnalyzer::TyreAnalyzer()
    : ocrEngine_(TesseractOcrEngine::Settings{}) {
}

AnalysisResult TyreAnalyzer::analyzeImageFile(const std::string& imagePath, const std::string& outputDir) {
    AnalysisResult result;
    result.inputPath = imagePath;

    const cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
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

AnalysisResult TyreAnalyzer::analyzeFrame(const cv::Mat& frame,
                                          const std::string& frameId,
                                          const std::string& outputDir) {
    AnalysisResult result;
    result.frameId = frameId;

    const Clock::time_point totalStart = Clock::now();
    fs::create_directories(outputDir);

    const std::string safeStem = makeSafeStem(frameId.empty() ? "frame" : frameId);
    const std::string sizeCropPath = (fs::path(outputDir) / (safeStem + "_size_crop.png")).string();
    const std::string dotCropPath = (fs::path(outputDir) / (safeStem + "_dot_crop.png")).string();
    const std::string overlayPath = (fs::path(outputDir) / (safeStem + "_overlay.png")).string();

    const Clock::time_point preprocessStart = Clock::now();
    const cv::Mat gray = preprocessor_.toGrayscale(frame);
    const cv::Mat deskewed = preprocessor_.deskewLight(gray);
    (void)preprocessor_.applyClahe(deskewed);
    result.timings.preprocessMs = elapsedMs(preprocessStart, Clock::now());

    const Clock::time_point roiStart = Clock::now();
    const std::vector<CandidateRoi> rois = preprocessor_.proposeTextRegions(frame);
    std::vector<cv::Rect> overlayBoxes;
    result.timings.roiProposalMs = elapsedMs(roiStart, Clock::now());
    if (rois.empty()) {
        result.notes.push_back("No ROI candidates found.");
    }

    const Clock::time_point ocrStart = Clock::now();
    const std::vector<CandidateText> candidateTexts = collectCandidateTexts(frame, rois, result, overlayBoxes);
    result.timings.ocrMs = elapsedMs(ocrStart, Clock::now());

    const Clock::time_point parsingStart = Clock::now();
    for (const auto& candidate : candidateTexts) {
        if (candidate.size.found) {
            const double confidence = computeSizeConfidence(candidate.size, candidate);
            if (confidence > result.tyreSize.confidence) {
                result.tyreSize.rawText = candidate.size.raw;
                result.tyreSize.normalizedText = candidate.size.normalized;
                result.tyreSize.found = true;
                result.tyreSize.confidence = confidence;
                result.tyreSize.uncertainty = 1.0 - confidence;
                result.tyreSize.roiQuality = candidate.roi.imageQualityScore;
                result.tyreSize.boundingBox = candidate.roi.box;
                result.tyreSizeFound = true;
            }
        }

        if (candidate.dot.dotFound || candidate.dot.weekYearFound) {
            const double confidence = computeDotConfidence(candidate.dot, candidate);
            if (confidence > result.dot.confidence) {
                result.dot.rawText = candidate.dot.raw;
                result.dot.normalizedText = candidate.dot.normalized;
                result.dot.found = candidate.dot.dotFound || candidate.dot.weekYearFound;
                result.dot.confidence = confidence;
                result.dot.uncertainty = 1.0 - confidence;
                result.dot.roiQuality = candidate.roi.imageQualityScore;
                result.dot.boundingBox = candidate.roi.box;

                result.dotFound = candidate.dot.dotFound || candidate.dot.weekYearFound;
                result.dotWeekYearFound = candidate.dot.weekYearFound;
                result.dotFullFound = candidate.dot.fullFound;
                result.dotWeekYear = candidate.dot.weekYear;
                result.dotFullRaw = candidate.dot.fullRaw;
                result.dotFullNormalized = candidate.dot.fullNormalized;
            }
        }
    }
    result.timings.parsingMs = elapsedMs(parsingStart, Clock::now());

    if (!result.tyreSizeFound) {
        result.notes.push_back("Tyre size not found in candidate ROIs.");
    }
    if (!result.dotFound) {
        result.notes.push_back("DOT code not found in candidate ROIs.");
    } else if (!result.dotFullFound) {
        result.notes.push_back("Only partial DOT information available.");
    }

    const Clock::time_point cropStart = Clock::now();
    saveCropOrPlaceholder(frame, result.tyreSize.boundingBox, sizeCropPath, "SIZE ROI NOT FOUND", result.tyreSizeFound);
    saveCropOrPlaceholder(frame, result.dot.boundingBox, dotCropPath, "DOT ROI NOT FOUND", result.dotFound);
    result.tyreSize.cropPath = sizeCropPath;
    result.dot.cropPath = dotCropPath;
    result.timings.cropSaveMs = elapsedMs(cropStart, Clock::now());

    const Clock::time_point overlayStart = Clock::now();
    saveOverlay(frame, overlayBoxes, result.tyreSize.boundingBox, result.dot.boundingBox, overlayPath);
    result.overlayPath = overlayPath;
    result.timings.overlaySaveMs = elapsedMs(overlayStart, Clock::now());

    result.timings.totalMs = elapsedMs(totalStart, Clock::now());
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

    const std::regex dotRegex(R"(DOT([A-Z0-9]{6,20}))");
    std::smatch dotMatch;
    if (std::regex_search(compactOriginal, dotMatch, dotRegex)) {
        result.dotFound = true;
        result.raw = squeezeSpaces(text);
        result.fullRaw = "DOT" + dotMatch[1].str();
        result.fullNormalized = result.fullRaw;
        result.normalized = result.fullNormalized;
        result.fullFound = dotMatch[1].str().size() >= 8;
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
    AnalysisResult& result,
    std::vector<cv::Rect>& overlayBoxes) const {
    std::vector<CandidateText> candidates;

    if (!ocrEngine_.isInitialized()) {
        result.notes.push_back("Tesseract OCR initialization failed. OCR results are unavailable.");
        return candidates;
    }

    for (const auto& roi : rois) {
        const cv::Rect bounded = roi.box & cv::Rect(0, 0, frame.cols, frame.rows);
        if (bounded.empty()) {
            continue;
        }
        if (bounded.width < 48 || bounded.height < 16) {
            continue;
        }

        const double aspectRatio = static_cast<double>(bounded.width) / std::max(1, bounded.height);
        if (aspectRatio < 1.2 || aspectRatio > 25.0) {
            continue;
        }

        overlayBoxes.push_back(bounded);
        const cv::Mat crop = frame(bounded).clone();
        const auto variants = preprocessor_.buildOcrVariants(crop);
        if (variants.empty()) {
            continue;
        }

        std::vector<OcrResult> blockResults = ocrEngine_.recognizeVariants(variants, tesseract::PSM_SINGLE_BLOCK);
        if (aspectRatio >= 4.0 && roi.imageQualityScore >= 0.20) {
            std::vector<OcrResult> lineResults = ocrEngine_.recognizeVariants(variants, tesseract::PSM_SINGLE_LINE);
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
            candidates.push_back(std::move(candidate));
        }
    }

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
