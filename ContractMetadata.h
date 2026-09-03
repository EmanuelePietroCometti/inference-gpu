#pragma once

#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

//
// ContractMetadata
// Ports the "shared contract" logic of ONNX_inference/AnomalyEngine into a
// reusable, batch-friendly form. Two responsibilities:
//
//  1. ContractMetadata : reads the calibration/preprocessing contract from the
//     ONNX model metadata (contract_version, normalize_inside_graph,
//     preproc_color_conversion, map_min_raw, map_max_raw, image_threshold_raw,
//     pixel_threshold_raw). Fails fast when the mandatory calibration keys are
//     missing, exactly like AnomalyEngine::LoadContractMetadata.
//
//  2. ResizeCoeffs + ResizeAntialias : the separable antialiased bilinear
//     resample that matches the Python reference bit-for-bit. Unlike the
//     original member-function version, the coefficients are a read-only value
//     object and the resize keeps its scratch buffer LOCAL, so the same coeffs
//     can be shared by every thread of the batch cv::parallel_for.
//

// Precomputed separable triangle-filter coefficients for a fixed src->dst size.
struct ResizeCoeffs {
    int inW = 0, inH = 0, outW = 0, outH = 0;
    int hK = 0, vK = 0;                        // taps per output pixel (H / V axis)
    std::vector<int> hBounds, vBounds;         // first source index per output pixel
    std::vector<double> hWeights, vWeights;    // [out_size * ksize] normalized weights

    // True when inW==outW && inH==outH. In that case the triangle filter
    // degenerates to weights [1, 0] on bounds[o]==o for EVERY output pixel, i.e.
    // the resample is a provably exact identity. ResizeAntialias then skips the
    // two O(W*H*taps) double-precision passes entirely (bit-identical output).
    bool identity = false;

    // Build coefficients for (inW x inH) -> (outW x outH). Call once per size.
    void Build(int inW, int inH, int outW, int outH);
    bool Matches(int srcW, int srcH) const { return srcW == inW && srcH == inH; }
};

// Antialiased bilinear resize of a CV_8UC3 image to (coeffs.outW x coeffs.outH).
// Thread-safe: reads only the shared coeffs and allocates its own intermediate
// buffer, so concurrent calls with the same coeffs never race.
// WARNING: on the identity fast path 'dst' is a SHALLOW alias of 'src' (shared
// pixel buffer, no copy). Callers must treat dst as READ-ONLY.
void ResizeAntialias(const ResizeCoeffs& coeffs, const cv::Mat& src, cv::Mat& dst);

// Contract parameters read from the ONNX model metadata.
struct ContractMetadata {
    bool  normalizationInGraph = false; // true -> model does /255+mean/std itself
    bool  convertBgrToRgb = true;       // swap channels before feeding the graph
    bool  hasPixelThreshold = false;
    float mapMin = 0.0f;
    float mapMax = 1.0f;
    float scoreThreshold = 0.5f;        // image_threshold_raw
    float pixelThreshold = 0.0f;        // pixel_threshold_raw
    std::string contractVersion;

    // Reads and validates the contract from an open session. Throws
    // std::runtime_error if the mandatory calibration keys are missing/invalid.
    void Load(Ort::Session& session);
};
