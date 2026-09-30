// Tests for Abbadon Engine: every check that engine.cpp prints by hand (and the ones
// left commented out there), plus the findings documented in .claude/Claude.md,
// turned into automatic PASS/FAIL checks.
//
// Run from the project root (paths are relative):
//   ./build/engine_tests
// or through CTest:
//   ctest --test-dir build --output-on-failure

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>

#include "abbadon/processors.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

const std::string DAOWA_PATH     = "weights/Daowa_Oracle_Frozen.pt";
const std::string MENDICANT_PATH = "weights/mendicant_bias_cpp.pt";
const std::string DOG_IMAGE      = "images/fredo.jpeg";   // Fredo  -> perro
const std::string CAT_IMAGE      = "images/masha.jpeg";   // Masha  -> gato

// Class order confirmed in mendicant_dataset.py: {'Cat': 0, 'Dog': 1}
const int64_t CAT = 0;
const int64_t DOG = 1;

int passed = 0;
int failed = 0;

void check(bool condition, const std::string& name) {
    if (condition) {
        ++passed;
        std::cout << "  [PASS] " << name << std::endl;
    } else {
        ++failed;
        std::cout << "  [FAIL] " << name << std::endl;
    }
}

bool has_shape(const torch::Tensor& t, const std::vector<int64_t>& expected) {
    return t.sizes().vec() == expected;
}

// Same preprocessing as engine.cpp (a copy for now).
// When it moves into processors.cpp during the refactor, call that function instead,
// so the tests check the REAL engine code and not a copy of it.
torch::Tensor preprocess(const cv::Mat& bgr, const c10::Device& device) {
    cv::Mat rgb, resized;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::resize(rgb, resized, cv::Size(384, 384), 0, 0, cv::INTER_LINEAR);

    // from_blob only watches `resized`, but .to(device) copies the data to the GPU,
    // so it's safe that `resized` dies when this function returns
    torch::Tensor hwc = torch::from_blob(resized.data, {384, 384, 3}, torch::kUInt8);
    torch::Tensor chw = hwc.to(device).permute({2, 0, 1}).to(torch::kFloat32).div(255.0f).contiguous();

    torch::Tensor mean  = torch::tensor({0.485f, 0.456f, 0.406f}, device).view({3, 1, 1});
    torch::Tensor stdev = torch::tensor({0.229f, 0.224f, 0.225f}, device).view({3, 1, 1});
    return ((chw - mean) / stdev).unsqueeze(0);
}

torch::Tensor run_daowa(torch::jit::script::Module& daowa, const torch::Tensor& input) {
    std::vector<torch::jit::IValue> inputs{input};
    return torch::sigmoid(daowa.forward(inputs).toTensor());
}

// ---------------------------------------------------------------------------

void test_cuda() {
    std::cout << "\n[CUDA]" << std::endl;
    check(torch::cuda::is_available(), "CUDA is available");
    check(torch::cuda::device_count() >= 1, "At least one GPU is visible");
}

void test_load_models(torch::jit::script::Module& daowa, torch::jit::script::Module& mendicant) {
    std::cout << "\n[load_model]" << std::endl;
    check(load_model(DAOWA_PATH, daowa), "Daowa-maad (Frozen) loads");
    check(load_model(MENDICANT_PATH, mendicant), "Mendicant Bias loads");

    // The error message printed below is expected: we are testing the failure path
    torch::jit::script::Module missing;
    check(!load_model("weights/does_not_exist.pt", missing), "load_model returns false for a missing file");
}

// The test that is commented out in engine.cpp (random tensor through Daowa)
void test_daowa_dummy(torch::jit::script::Module& daowa, const c10::Device& device) {
    std::cout << "\n[Daowa-maad with random input]" << std::endl;
    torch::Tensor dummy = torch::rand({1, 3, 384, 384}, device);
    std::vector<torch::jit::IValue> inputs{dummy};
    torch::Tensor out = daowa.forward(inputs).toTensor();
    check(has_shape(out, {1, 1, 384, 384}), "Output shape is [1, 1, 384, 384]");
}

void test_preprocessing(const c10::Device& device) {
    std::cout << "\n[Preprocessing: " << DOG_IMAGE << "]" << std::endl;

    cv::Mat bgr = cv::imread(DOG_IMAGE);
    check(!bgr.empty(), "Image loads");
    if (bgr.empty()) {
        return;
    }
    check(bgr.channels() == 3, "Image has 3 channels");

    // BGR -> RGB really swaps channels 0 and 2
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    cv::Vec3b px_bgr = bgr.at<cv::Vec3b>(100, 100);
    cv::Vec3b px_rgb = rgb.at<cv::Vec3b>(100, 100);
    check(px_rgb[0] == px_bgr[2] && px_rgb[2] == px_bgr[0], "cvtColor swaps B and R");

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(384, 384), 0, 0, cv::INTER_LINEAR);
    check(resized.rows == 384 && resized.cols == 384 && resized.channels() == 3, "Resized to 384x384x3");

    // from_blob: shape, dtype, and that it reads the RESIZED image
    // (regression for the bug where from_blob read the 1600x900 image)
    torch::Tensor hwc = torch::from_blob(resized.data, {384, 384, 3}, torch::kUInt8);
    check(has_shape(hwc, {384, 384, 3}), "from_blob shape is [384, 384, 3]");
    check(hwc.scalar_type() == torch::kUInt8, "from_blob dtype is uint8");
    bool same_pixels = true;
    for (int y : {0, 150, 383}) {
        for (int x : {0, 200, 383}) {
            for (int c = 0; c < 3; c++) {
                if (hwc[y][x][c].item<uint8_t>() != resized.at<cv::Vec3b>(y, x)[c]) {
                    same_pixels = false;
                }
            }
        }
    }
    check(same_pixels, "Tensor pixels match the resized cv::Mat pixels");

    // The contiguity test that is commented out in engine.cpp
    torch::Tensor permuted = hwc.to(device).permute({2, 0, 1}).to(torch::kFloat32).div(255.0f);
    check(!permuted.is_contiguous(), "permute leaves the tensor non-contiguous");
    torch::Tensor chw = permuted.contiguous();
    check(chw.is_contiguous(), ".contiguous() makes it contiguous");
    check(chw.device().is_cuda(), "Tensor is on the GPU");
    check(chw.scalar_type() == torch::kFloat32, "Tensor dtype is float32");
    check(has_shape(chw, {3, 384, 384}), "Tensor shape is [3, 384, 384]");
    check(chw.min().item<float>() >= 0.0f && chw.max().item<float>() <= 1.0f, "Values in [0, 1] after /255");

    // Full preprocessing: ImageNet normalization + batch dimension.
    // Theoretical range: (0 - 0.485) / 0.229 = -2.118 ... (1 - 0.406) / 0.225 = 2.64
    torch::Tensor input = preprocess(bgr, device);
    check(has_shape(input, {1, 3, 384, 384}), "Input shape is [1, 3, 384, 384]");
    float mn = input.min().item<float>();
    float mx = input.max().item<float>();
    std::cout << "         min " << mn << " | max " << mx << std::endl;
    check(mn >= -2.1179f - 1e-3f && mx <= 2.6400f + 1e-3f, "Normalized values inside the ImageNet range");
    check(mn < 0.0f && mx > 1.0f, "Normalization was applied (values leave [0, 1])");
}

void test_daowa_real(torch::jit::script::Module& daowa, const c10::Device& device) {
    std::cout << "\n[Daowa-maad with " << DOG_IMAGE << "]" << std::endl;
    cv::Mat bgr = cv::imread(DOG_IMAGE);
    if (bgr.empty()) {
        check(false, "Image loads");
        return;
    }
    torch::Tensor probs = run_daowa(daowa, preprocess(bgr, device));

    check(has_shape(probs, {1, 1, 384, 384}), "Mask shape is [1, 1, 384, 384]");
    check(probs.min().item<float>() >= 0.0f && probs.max().item<float>() <= 1.0f, "Probabilities in [0, 1]");

    // A real segmentation marks part of the image: ~0% or ~100% means it sees nothing / everything
    float pet = (probs > 0.5f).to(torch::kFloat32).mean().item<float>();
    std::cout << "         pet pixels " << pet * 100.0f << "%" << std::endl;
    check(pet > 0.05f && pet < 0.95f, "Pet pixels between 5% and 95%");

    // Visualization conversion (regression for .mul(255.0f) promoting uint8 -> float)
    torch::Tensor mask_cpu = (probs > 0.5f).to(torch::kUInt8).mul(255).squeeze().to(torch::kCPU).contiguous();
    check(mask_cpu.scalar_type() == torch::kUInt8, "Mask stays uint8 after .mul(255)");
    check(has_shape(mask_cpu, {384, 384}), "Mask for OpenCV is [384, 384]");
    check(((mask_cpu == 0) | (mask_cpu == 255)).all().item<bool>(), "Mask only has 0 and 255");
}

void test_mendicant(torch::jit::script::Module& daowa, torch::jit::script::Module& mendicant,
                    const c10::Device& device, const std::string& image, int64_t expected) {
    const std::array<std::string, 2> names = {"gato", "perro"};
    std::cout << "\n[Mendicant Bias with " << image << " -> " << names[expected] << "]" << std::endl;

    cv::Mat bgr = cv::imread(image);
    if (bgr.empty()) {
        check(false, "Image loads");
        return;
    }
    torch::Tensor input = preprocess(bgr, device);
    torch::Tensor probs = run_daowa(daowa, input);

    // forward(rgb, daowa_mask): same normalized RGB + sigmoid WITHOUT threshold
    std::vector<torch::jit::IValue> inputs{input, probs};
    torch::Tensor logits = mendicant.forward(inputs).toTensor();
    check(has_shape(logits, {1, 2}), "Logits shape is [1, 2]");

    torch::Tensor conf = torch::softmax(logits, 1);
    check(std::abs(conf.sum().item<float>() - 1.0f) < 1e-4f, "Softmax sums to 1");

    int64_t predicted = logits.argmax(1).item<int64_t>();
    std::cout << "         " << names[predicted] << " " << conf[0][predicted].item<float>() * 100.0f << "%" << std::endl;
    check(predicted == expected, "Predicts " + names[expected]);
}

}  // namespace

int main() {
    torch::NoGradGuard no_grad;
    c10::Device device = get_device();

    test_cuda();

    torch::jit::script::Module daowa;
    torch::jit::script::Module mendicant;
    test_load_models(daowa, mendicant);
    if (failed > 0) {
        std::cout << "\nModels didn't load, skipping the rest." << std::endl;
        return 1;
    }
    daowa.eval();
    mendicant.eval();

    test_daowa_dummy(daowa, device);
    test_preprocessing(device);
    test_daowa_real(daowa, device);
    test_mendicant(daowa, mendicant, device, DOG_IMAGE, DOG);
    test_mendicant(daowa, mendicant, device, CAT_IMAGE, CAT);

    std::cout << "\n==========================================" << std::endl;
    std::cout << passed << " passed, " << failed << " failed" << std::endl;
    return failed == 0 ? 0 : 1;
}
