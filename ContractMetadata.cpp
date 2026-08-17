#include "ContractMetadata.h"
#include "AsyncLogger.h"

#include <cmath>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Separable antialiased bilinear resample. Mirrors
// inference_simulation/src/antialias_resize.py::precompute_coeffs bit-for-bit:
// on a downscale the filter support widens by the scale factor (antialiasing).
// This is the SAME algorithm as ONNX_inference/AnomalyEngine, refactored so the
// coefficients are a shareable value object and the resize scratch is local.
// ---------------------------------------------------------------------------

namespace {

void PrecomputeCoeffs(int inSize, int outSize,
    std::vector<int>& bounds, std::vector<double>& weights, int& ksize)
{
    const double scale = static_cast<double>(inSize) / static_cast<double>(outSize);
    const double filterscale = scale >= 1.0 ? scale : 1.0;   // widen on downscale
    const double support = 1.0 * filterscale;                // bilinear half-width
    const double ss = 1.0 / filterscale;

    ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    bounds.assign(outSize, 0);
    weights.assign(static_cast<size_t>(outSize) * ksize, 0.0);

    for (int o = 0; o < outSize; ++o) {
        const double center = (o + 0.5) * scale;
        int xmin = static_cast<int>(center - support + 0.5);
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > inSize) xmax = inSize;
        const int n = xmax - xmin;
        double total = 0.0;
        for (int t = 0; t < n; ++t) {
            double w = 1.0 - std::abs((xmin + t - center + 0.5) * ss);  // bilinear_filter
            if (w < 0.0) w = 0.0;
            weights[static_cast<size_t>(o) * ksize + t] = w;
            total += w;
        }
        if (total > 0.0)
            for (int t = 0; t < n; ++t) weights[static_cast<size_t>(o) * ksize + t] /= total;
        bounds[o] = xmin;
    }
}

inline uint8_t RoundClipU8(double v) {
    double r = std::floor(v + 0.5);   // round half up (pixels are non-negative)
    if (r < 0.0) r = 0.0;
    if (r > 255.0) r = 255.0;
    return static_cast<uint8_t>(r);
}

} // namespace

void ResizeCoeffs::Build(int inW_, int inH_, int outW_, int outH_)
{
    inW = inW_; inH = inH_; outW = outW_; outH = outH_;
    PrecomputeCoeffs(inW, outW, hBounds, hWeights, hK);
    PrecomputeCoeffs(inH, outH, vBounds, vWeights, vK);
}

void ResizeAntialias(const ResizeCoeffs& c, const cv::Mat& src, cv::Mat& dst)
{
    const int inH = src.rows, inW = src.cols;
    const int outW = c.outW, outH = c.outH;

    // Horizontal pass: [inH x inW] -> [inH x outW]. Scratch is LOCAL, so this
    // function is safe to call concurrently from the batch parallel_for.
    cv::Mat hpass(inH, outW, CV_8UC3);
    for (int y = 0; y < inH; ++y) {
        const uint8_t* srow = src.ptr<uint8_t>(y);
        uint8_t* hrow = hpass.ptr<uint8_t>(y);
        for (int o = 0; o < outW; ++o) {
            const int s = c.hBounds[o];
            const int avail = (std::min)(c.hK, inW - s);
            const double* w = &c.hWeights[static_cast<size_t>(o) * c.hK];
            double a0 = 0.0, a1 = 0.0, a2 = 0.0;
            for (int t = 0; t < avail; ++t) {
                const uint8_t* px = srow + static_cast<size_t>(s + t) * 3;
                const double wt = w[t];
                a0 += wt * px[0]; a1 += wt * px[1]; a2 += wt * px[2];
            }
            uint8_t* op = hrow + static_cast<size_t>(o) * 3;
            op[0] = RoundClipU8(a0); op[1] = RoundClipU8(a1); op[2] = RoundClipU8(a2);
        }
    }

    // Vertical pass: [inH x outW] -> [outH x outW]
    dst.create(outH, outW, CV_8UC3);
    for (int o = 0; o < outH; ++o) {
        const int s = c.vBounds[o];
        const int avail = (std::min)(c.vK, inH - s);
        const double* w = &c.vWeights[static_cast<size_t>(o) * c.vK];
        uint8_t* drow = dst.ptr<uint8_t>(o);
        for (int x = 0; x < outW; ++x) {
            double a0 = 0.0, a1 = 0.0, a2 = 0.0;
            for (int t = 0; t < avail; ++t) {
                const uint8_t* px = hpass.ptr<uint8_t>(s + t) + static_cast<size_t>(x) * 3;
                const double wt = w[t];
                a0 += wt * px[0]; a1 += wt * px[1]; a2 += wt * px[2];
            }
            uint8_t* op = drow + static_cast<size_t>(x) * 3;
            op[0] = RoundClipU8(a0); op[1] = RoundClipU8(a1); op[2] = RoundClipU8(a2);
        }
    }
}

// ---------------------------------------------------------------------------
// Contract metadata loader. 1:1 with AnomalyEngine::LoadContractMetadata.
// ---------------------------------------------------------------------------

void ContractMetadata::Load(Ort::Session& session)
{
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::ModelMetadata metadata = session.GetModelMetadata();

    auto lookupString = [&](const char* key) -> std::string {
        auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
        return value ? std::string(value.get()) : std::string();
        };
    auto lookupFloat = [&](const char* key, float& target) -> bool {
        auto value = metadata.LookupCustomMetadataMapAllocated(key, allocator);
        if (value) {
            target = std::stof(value.get());
            Log::Info("Metadata '{}' = {}", key, target);
            return true;
        }
        return false;
        };

    contractVersion = lookupString("contract_version");
    if (contractVersion.empty()) {
        Log::Warning("Metadata 'contract_version' missing. Proceeding assuming backwards compatibility.");
    }
    else {
        Log::Info("ONNX Contract Version validated: {}", contractVersion);
    }

    const std::string normalization = lookupString("normalize_inside_graph");
    normalizationInGraph = (normalization == "true");
    if (!normalizationInGraph) {
        Log::Warning("### WARNING: normalize_inside_graph='{}' (expected 'true'). Host will apply ImageNet normalization as fallback.", normalization);
    }

    // Missing key defaults to bgr2rgb (all our exporter's models).
    const std::string colorConversion = lookupString("preproc_color_conversion");
    convertBgrToRgb = colorConversion.empty() || colorConversion == "bgr2rgb";
    Log::Info("Metadata 'preproc_color_conversion' = '{}' -> BGR2RGB swap {}.",
        colorConversion.empty() ? "(missing, default)" : colorConversion,
        convertBgrToRgb ? "ENABLED" : "disabled");

    const std::string calibratedStatus = lookupString("calibrated");
    if (calibratedStatus != "true") {
        Log::Warning("Model 'calibrated' flag is not true. Using uncalibrated raw tensor bounds.");
    }

    const bool hasMin = lookupFloat("map_min_raw", mapMin);
    const bool hasMax = lookupFloat("map_max_raw", mapMax);
    const bool hasThreshold = lookupFloat("image_threshold_raw", scoreThreshold);
    hasPixelThreshold = lookupFloat("pixel_threshold_raw", pixelThreshold);

    if (!hasMin || !hasMax || !hasThreshold) {
        throw std::runtime_error(
            "Anomaly model rejected: calibration metadata is incomplete "
            "(map_min_raw/map_max_raw/image_threshold_raw). Re-run calibrate.py.");
    }
    if (mapMax - mapMin <= 0.0f) {
        throw std::runtime_error("Invalid calibration metadata: map_max_raw must be greater than map_min_raw.");
    }

    const std::string normFormula = lookupString("normalization_formula");
    if (!normFormula.empty() && normFormula != "minmax") {
        Log::Warning("Model exported with normalization_formula='{}'. C++ engine uses 'minmax' natively.", normFormula);
    }

    Log::Info("Contract limits verified. mapMin={}, mapMax={}, scoreThreshold={}, pixelThreshold={} (present={}), normalizationInGraph={}.",
        mapMin, mapMax, scoreThreshold, pixelThreshold, hasPixelThreshold, normalizationInGraph);
}
