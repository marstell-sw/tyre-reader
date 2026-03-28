#pragma once

#include "Types.h"

#include <opencv2/core.hpp>
#include <tesseract/baseapi.h>

#include <string>
#include <utility>
#include <vector>

namespace tyre {

class TesseractOcrEngine {
public:
    struct Settings {
        std::string language = "eng";
        std::string dataPath;
        tesseract::OcrEngineMode engineMode = tesseract::OEM_LSTM_ONLY;
        tesseract::PageSegMode defaultPageSegMode = tesseract::PSM_SINGLE_BLOCK;
    };

    TesseractOcrEngine();
    explicit TesseractOcrEngine(const Settings& settings);
    ~TesseractOcrEngine();

    TesseractOcrEngine(const TesseractOcrEngine&) = delete;
    TesseractOcrEngine& operator=(const TesseractOcrEngine&) = delete;

    bool isInitialized() const;

    OcrResult recognize(const cv::Mat& image,
                        const std::string& variantName = "default",
                        tesseract::PageSegMode pageSegMode = tesseract::PSM_SINGLE_BLOCK,
                        const std::string& whitelist = "") const;

    std::vector<OcrResult> recognizeVariants(
        const std::vector<std::pair<std::string, cv::Mat>>& variants,
        tesseract::PageSegMode pageSegMode = tesseract::PSM_SINGLE_BLOCK,
        const std::string& whitelist = "") const;

private:
    Settings settings_;
    mutable tesseract::TessBaseAPI api_;
    bool initialized_ = false;

    static std::string resolveDataPath(const Settings& settings);
    static std::string trim(const std::string& value);
};

}  // namespace tyre
