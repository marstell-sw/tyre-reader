#pragma once

#include "ImagePreprocessor.h"
#include "TesseractOcrEngine.h"
#include "Types.h"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace tyre {

class TyreAnalyzer {
public:
    explicit TyreAnalyzer(bool saveDebugArtifacts = false);

    AnalysisResult analyzeImageFile(const std::string& imagePath, const std::string& outputDir);
    std::vector<AnalysisResult> analyzeDirectory(const std::string& inputDir, const std::string& outputDir);
    AnalysisResult analyzeFrame(const cv::Mat& frame, const std::string& frameId, const std::string& outputDir);
    WheelExtractionResult extractWheelGeometryFile(const std::string& imagePath, const std::string& outputDir);
    std::vector<WheelExtractionResult> extractWheelGeometryDirectory(const std::string& inputDir,
                                                                    const std::string& outputDir);

private:
    struct ParsedSize {
        bool found = false;
        std::string raw;
        std::string normalized;
        double parseQuality = 0.0;
    };

    struct ParsedDot {
        bool dotFound = false;
        bool weekYearFound = false;
        bool fullFound = false;
        std::string raw;
        std::string normalized;
        std::string weekYear;
        std::string fullRaw;
        std::string fullNormalized;
        double parseQuality = 0.0;
    };

    struct CandidateText {
        CandidateRoi roi;
        OcrResult ocr;
        ParsedSize size;
        ParsedDot dot;
    };

    struct OcrProbe {
        std::string name;
        cv::Rect roi;
        cv::Mat image;
    };

    ImagePreprocessor preprocessor_;
    TesseractOcrEngine ocrEngine_;
    bool saveDebugArtifacts_ = false;

    static std::string makeSafeStem(const std::string& value);
    static std::string squeezeSpaces(const std::string& value);
    static std::string sanitizeForParsing(const std::string& value);
    static std::string normalizeDotToken(const std::string& value);
    static bool hasSupportedImageExtension(const std::string& extension);
    static void appendTiming(std::vector<NamedTiming>& timings, const std::string& name, double ms);
    static void saveDebugImage(const cv::Mat& image, const std::string& path);
    static void writeDebugReport(const std::string& path, const std::vector<NamedTiming>& timings);
    static std::string sanitizeCsvField(const std::string& value);
    WheelExtractionResult extractWheelGeometryFrame(const cv::Mat& frame,
                                                   const std::string& frameId,
                                                   const std::string& inputPath,
                                                   const std::string& outputDir) const;
    std::vector<OcrProbe> buildStripProbes(const cv::Mat& image, const std::string& prefix) const;
    std::vector<std::pair<std::string, cv::Mat>> buildFastVariants(const cv::Mat& image, bool aggressiveThreshold) const;
    FieldResult detectTyreSizeField(const cv::Mat& image, const std::string& debugDir, AnalysisResult& result) const;
    FieldResult detectDotField(const cv::Mat& image, const std::string& debugDir, AnalysisResult& result) const;

    ParsedSize parseTyreSize(const std::string& text) const;
    ParsedDot parseDot(const std::string& text) const;
    double computeSizeConfidence(const ParsedSize& parsed, const CandidateText& candidate) const;
    double computeDotConfidence(const ParsedDot& parsed, const CandidateText& candidate) const;
    std::vector<CandidateText> collectCandidateTexts(const cv::Mat& frame,
                                                     const std::vector<CandidateRoi>& rois,
                                                     const std::string& debugDir,
                                                     AnalysisResult& result,
                                                     std::vector<cv::Rect>& overlayBoxes) const;

    void saveCropOrPlaceholder(const cv::Mat& source,
                               const cv::Rect& roi,
                               const std::string& path,
                               const std::string& label,
                               bool roiFound) const;

    void saveOverlay(const cv::Mat& image,
                     const std::vector<cv::Rect>& roiBoxes,
                     const cv::Rect& sizeBox,
                     const cv::Rect& dotBox,
                     const std::string& path) const;
};

}  // namespace tyre
