#pragma once

#include "ImagePreprocessor.h"
#include "TesseractOcrEngine.h"
#include "Types.h"

#include <opencv2/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace tyre {

class TyreAnalyzer {
public:
    explicit TyreAnalyzer(bool saveDebugArtifacts = false, bool skipOcr = false);
    ~TyreAnalyzer();

    AnalysisResult analyzeImageFile(const std::string& imagePath, const std::string& outputDir);
    std::vector<AnalysisResult> analyzeDirectory(const std::string& inputDir, const std::string& outputDir);
    AnalysisResult analyzeFrame(const cv::Mat& frame, const std::string& frameId, const std::string& outputDir);
    RoiOcrResult recognizeRoiFile(const std::string& imagePath,
                                  const cv::Rect& roi,
                                  const std::string& branch,
                                  const std::string& outputDir);
    SectorUnwrapResult unwrapSectorFile(const std::string& imagePath,
                                        const std::string& branch,
                                        double startAngleDeg,
                                        double endAngleDeg,
                                        bool useWheelOverride,
                                        const cv::Point2f& wheelCenter,
                                        float wheelInnerRadius,
                                        float wheelOuterRadius,
                                        const std::string& outputDir);
    WheelExtractionResult extractWheelGeometryFile(const std::string& imagePath, const std::string& outputDir);
    std::vector<WheelExtractionResult> extractWheelGeometryDirectory(const std::string& inputDir,
                                                                    const std::string& outputDir);

private:
    struct YoloRuntime;

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

    struct StripCandidate {
        std::string name;
        cv::Rect extendedBox;
        cv::Rect mappedBox;
        cv::Mat image;
        double geometryScore = 0.0;
        double imageQualityScore = 0.0;
    };

    struct YoloPredictionRun {
        bool ok = false;
        std::string overlayPath;
        double elapsedMs = 0.0;
        std::vector<YoloDetection> detections;
        std::vector<std::string> notes;
    };

    struct AnnulusLocalRoi {
        cv::Mat image;
        cv::Rect bandRect;
        double startAngleDeg = 0.0;
        double endAngleDeg = 0.0;
        double radiusMin = 0.0;
        double radiusMax = 0.0;
    };

    ImagePreprocessor preprocessor_;
    TesseractOcrEngine ocrEngine_;
    bool saveDebugArtifacts_ = false;
    bool skipOcr_ = false;
    mutable std::unique_ptr<YoloRuntime> yoloRuntime_;

    static std::string makeSafeStem(const std::string& value);
    static std::string squeezeSpaces(const std::string& value);
    static std::string sanitizeForParsing(const std::string& value);
    static std::string normalizeDotToken(const std::string& value);
    static bool hasSupportedImageExtension(const std::string& extension);
    static void appendTiming(std::vector<NamedTiming>& timings, const std::string& name, double ms);
    static void saveDebugImage(const cv::Mat& image, const std::string& path);
    static void writeDebugReport(const std::string& path, const std::vector<NamedTiming>& timings);
    static std::string sanitizeCsvField(const std::string& value);
    static cv::Rect clampRect(const cv::Rect& rect, const cv::Size& bounds);
    static std::pair<double, double> computeAngleRangeDeg(const cv::Rect& rect, const cv::Point2f& center);
    static double computeAnnulusCompatibility(const cv::Rect& rect, const ImagePreprocessor::WheelGeometry& geometry);
    static std::pair<double, double> computeRadiusRange(const cv::Rect& rect,
                                                        const cv::Point2f& center,
                                                        const ImagePreprocessor::WheelGeometry& geometry,
                                                        bool preferNarrowBand);
    static cv::Rect mapPolarWindowToBandRect(const cv::Size& sidewallBandSize,
                                             const ImagePreprocessor::WheelGeometry& geometry,
                                             double radiusMin,
                                             double radiusMax);
    static AnnulusLocalRoi extractLocalAnnulusRoi(const cv::Mat& sidewallBand,
                                                  const ImagePreprocessor::WheelGeometry& geometry,
                                                  const cv::Rect& rect,
                                                  bool preferNarrowBand);
    YoloPredictionRun runYoloRoiDetector(const cv::Mat& image,
                                         const std::string& frameId,
                                         const std::string& debugDir) const;
    WheelExtractionResult extractWheelGeometryFrame(const cv::Mat& frame,
                                                   const std::string& frameId,
                                                   const std::string& inputPath,
                                                   const std::string& outputDir) const;
    std::vector<OcrProbe> buildStripProbes(const cv::Mat& image, const std::string& prefix) const;
    std::vector<std::pair<std::string, cv::Mat>> buildFastVariants(const cv::Mat& image, bool aggressiveThreshold) const;
    std::vector<StripCandidate> detectSidewallCandidates(const cv::Mat& stripImage,
                                                         const std::string& fieldName,
                                                         const std::string& debugDir,
                                                         AnalysisResult& result) const;
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
