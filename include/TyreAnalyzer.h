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
    TyreAnalyzer();

    AnalysisResult analyzeImageFile(const std::string& imagePath, const std::string& outputDir);
    std::vector<AnalysisResult> analyzeDirectory(const std::string& inputDir, const std::string& outputDir);
    AnalysisResult analyzeFrame(const cv::Mat& frame, const std::string& frameId, const std::string& outputDir);

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

    ImagePreprocessor preprocessor_;
    TesseractOcrEngine ocrEngine_;

    static std::string makeSafeStem(const std::string& value);
    static std::string squeezeSpaces(const std::string& value);
    static std::string sanitizeForParsing(const std::string& value);
    static std::string normalizeDotToken(const std::string& value);
    static bool hasSupportedImageExtension(const std::string& extension);

    ParsedSize parseTyreSize(const std::string& text) const;
    ParsedDot parseDot(const std::string& text) const;
    double computeSizeConfidence(const ParsedSize& parsed, const CandidateText& candidate) const;
    double computeDotConfidence(const ParsedDot& parsed, const CandidateText& candidate) const;
    std::vector<CandidateText> collectCandidateTexts(const cv::Mat& frame,
                                                     const std::vector<CandidateRoi>& rois,
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
