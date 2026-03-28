#include "TesseractOcrEngine.h"

#include <leptonica/allheaders.h>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <vector>

namespace tyre {

namespace {

namespace fs = std::filesystem;

struct PixDestroyDeleter {
    void operator()(Pix* pix) const {
        if (pix != nullptr) {
            pixDestroy(&pix);
        }
    }
};

Pix* matToPix(const cv::Mat& image) {
    cv::Mat rgb;
    if (image.channels() == 1) {
        cv::cvtColor(image, rgb, cv::COLOR_GRAY2RGB);
    } else {
        cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);
    }

    Pix* pix = pixCreate(rgb.cols, rgb.rows, 32);
    if (pix == nullptr) {
        return nullptr;
    }

    for (int y = 0; y < rgb.rows; ++y) {
        l_uint32* line = pixGetData(pix) + static_cast<std::size_t>(y) * pixGetWpl(pix);
        for (int x = 0; x < rgb.cols; ++x) {
            const cv::Vec3b pixel = rgb.at<cv::Vec3b>(y, x);
            line[x] = (static_cast<l_uint32>(pixel[0]) << 24U) |
                      (static_cast<l_uint32>(pixel[1]) << 16U) |
                      (static_cast<l_uint32>(pixel[2]) << 8U);
        }
    }

    pixSetSpp(pix, 3);
    return pix;
}

}  // namespace

TesseractOcrEngine::TesseractOcrEngine()
    : TesseractOcrEngine(Settings{}) {
}

TesseractOcrEngine::TesseractOcrEngine(const Settings& settings)
    : settings_(settings) {
    settings_.dataPath = resolveDataPath(settings_);
    const char* dataPath = settings_.dataPath.empty() ? nullptr : settings_.dataPath.c_str();
    initialized_ = api_.Init(dataPath, settings_.language.c_str(), settings_.engineMode) == 0;
    if (initialized_) {
        api_.SetPageSegMode(settings_.defaultPageSegMode);
        api_.SetVariable("user_defined_dpi", "300");
        api_.SetVariable("preserve_interword_spaces", "1");
    }
}

TesseractOcrEngine::~TesseractOcrEngine() {
    if (initialized_) {
        api_.End();
    }
}

bool TesseractOcrEngine::isInitialized() const {
    return initialized_;
}

OcrResult TesseractOcrEngine::recognize(const cv::Mat& image,
                                        const std::string& variantName,
                                        tesseract::PageSegMode pageSegMode) const {
    OcrResult result;
    result.variantName = variantName;

    if (!initialized_ || image.empty()) {
        return result;
    }

    std::unique_ptr<Pix, PixDestroyDeleter> pix(matToPix(image));
    if (!pix) {
        return result;
    }

    api_.SetPageSegMode(pageSegMode);
    api_.SetImage(pix.get());

    std::unique_ptr<char[], decltype(&std::free)> text(api_.GetUTF8Text(), &std::free);
    result.text = trim(text ? std::string(text.get()) : std::string());
    result.averageConfidence = clamp01(static_cast<double>(api_.MeanTextConf()) / 100.0);
    api_.Clear();
    return result;
}

std::vector<OcrResult> TesseractOcrEngine::recognizeVariants(
    const std::vector<std::pair<std::string, cv::Mat>>& variants,
    tesseract::PageSegMode pageSegMode) const {
    std::vector<OcrResult> results;
    results.reserve(variants.size());
    for (const auto& variant : variants) {
        results.push_back(recognize(variant.second, variant.first, pageSegMode));
    }
    return results;
}

std::string TesseractOcrEngine::resolveDataPath(const Settings& settings) {
    auto hasLanguageData = [&](const fs::path& candidate) {
        return fs::exists(candidate / (settings.language + ".traineddata"));
    };

    auto resolveCandidate = [&](const fs::path& candidate) -> std::string {
        if (candidate.empty()) {
            return {};
        }
        if (hasLanguageData(candidate)) {
            return candidate.string();
        }
        if (hasLanguageData(candidate / "tessdata")) {
            return (candidate / "tessdata").string();
        }
        return {};
    };

    if (!settings.dataPath.empty()) {
        return resolveCandidate(fs::path(settings.dataPath));
    }

    std::vector<fs::path> candidates;
    if (const char* envValue = std::getenv("TESSDATA_PREFIX")) {
        candidates.emplace_back(envValue);
    }

    candidates.emplace_back("/usr/share/tesseract-ocr/5/tessdata");
    candidates.emplace_back("/usr/share/tesseract-ocr/4.00/tessdata");
    candidates.emplace_back("/usr/share/tessdata");
    candidates.emplace_back("/usr/local/share/tessdata");
    candidates.emplace_back(fs::current_path() / "tessdata");
    candidates.emplace_back(fs::current_path() / "third_party" / "tessdata");

    for (const auto& candidate : candidates) {
        const std::string resolved = resolveCandidate(candidate);
        if (!resolved.empty()) {
            return resolved;
        }
    }

    return {};
}

std::string TesseractOcrEngine::trim(const std::string& value) {
    const auto first = std::find_if_not(value.begin(), value.end(), [](unsigned char c) {
        return std::isspace(c) != 0;
    });
    const auto last = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) {
        return std::isspace(c) != 0;
    }).base();
    if (first >= last) {
        return {};
    }
    return std::string(first, last);
}

}  // namespace tyre
