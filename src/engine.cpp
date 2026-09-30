#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include "abbadon/processors.hpp"
#include "abbadon/check.hpp"

#include <iostream>
#include <array>
#include <string>
#include <vector>
#include <filesystem>

void check_shape(const cv::Mat& image);

// Main function
int main(int argc, char* argv[]) {

    // Usage: ./build/engine <image_path>
    // ALWAYS check argc before touching argv (argv[1] would be nullptr, argv[2] garbage)
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return 1;
    }
    std::string image_path = argv[1];

    // Output files next to the input: images/masha.jpeg -> images/masha_mask.png, images/masha_overlay.png
    std::filesystem::path input_path(image_path);
    std::string stem = input_path.stem().string();   // "masha"
    std::string mask_path    = (input_path.parent_path() / (stem + "_mask.png")).string();
    std::string overlay_path = (input_path.parent_path() / (stem + "_overlay.png")).string();

    // Declaring variables
    torch::jit::script::Module Daowa;
    torch::jit::script::Module Mendicant;
    std::array<std::string, 2> outputs = {"gato", "perro"};

    // Declared destination GPU
    auto device = get_device();

    // Check CUDA
    check_cuda();

    // Loading the model "Daowa"
    if (!load_model("weights/Daowa_Oracle_Frozen.pt", Daowa)) {
        return 1;
    }
    if (!load_model("weights/mendicant_bias_cpp.pt", Mendicant)) {
        return 1;
    }

    // Set the model in evaluation mode
    Daowa.eval();
    Mendicant.eval();

    /*
    
    THIS CODE WAS A TEST -----------------------------------------------------------------------------------------------

    // This is similar to with torch.no_grad():, but in C++. While this variable exists the model won't save gradients.
    // Note: This notation is RAII convention
    torch::NoGradGuard no_grad;

    // Declaring a dummy tensor to the model and it's output shape
    torch::Tensor dummy = torch::rand({1, 3, 384, 384}, device);

    // Now we've to packing this tensor in an IValue
    // because IValue it's like a box where I can save anything... maybe a tensor, a number, etc.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(dummy);

    // Run the model and print the output shape
    auto result = Daowa.forward(inputs);
    std::cout << "Output shape (It has to be: {1, 1, 384, 384}): " << result.toTensor().sizes() << std::endl;
    */
    
    // We load the image from the directory "images/"
    cv::Mat img = cv::imread(image_path);

    // Verify if the image isn't empty
    if (img.empty()) {
        std::cerr << "The image didn't load correctly, check the path, or perhaps the image doesn't exist." << std::endl;
        return 1;
    }

    // This function checks the image shape
    check_shape(img);

    // Declare a variable with the name "imgRGB" and this variable is used like output in the method cvtColor
    // all of this just to change the order BGR to RGB, cuz OpenCV loads the image in BGR format
    cv::Mat imgRGB;
    cv::cvtColor(img, imgRGB, cv::COLOR_BGR2RGB);

    // Resize the image
    cv::resize(imgRGB, img, cv::Size(384, 384), 0, 0, cv::INTER_LINEAR);

    // Check the shape of the new image
    std::cout << "------------------------------------------" << std::endl;
    check_shape(img);

    // Now we've to pass this image to Tensor in TorchLib
    // we'll use torch::from_blob, but from_blob just sees the memory, it doesn't own it.
    // The owner is the cv::Mat: if the Mat dies (scope or however), the tensor
    // will keep pointing to freed memory and this is dangerous
    torch::Tensor image_tensor = torch::from_blob(img.data, {384, 384, 3}, torch::kUInt8);

    std::cout << "Tensor shape: " << image_tensor.sizes()
            << "\nData type: " << image_tensor.dtype() << std::endl;

    torch::Tensor tensor_gpu = image_tensor.to(device).permute({2, 0, 1}).to(torch::kFloat32).div(255.0f).contiguous();

    std::cout << "Where is the tensor: " << tensor_gpu.device() << " and it's dtype: " << tensor_gpu.dtype() <<std::endl;
    std::cout << "Tensor's size: " << tensor_gpu.sizes() << std::endl;

    /*
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Tensor test (Is contiguous?)\n" 
            << "Tensor before: " << tensor_gpu.is_contiguous() << std::endl;
        
    tensor_gpu = tensor_gpu.contiguous();

    std::cout << "Tensor after: " << tensor_gpu.is_contiguous() << std::endl;
    */


    // ImageNet normalization: x = (x - mean) / std, per channel.
    // mean/std are created on the GPU with shape [3] and reshaped to [3, 1, 1],
    // so broadcasting pairs each value with ITS channel: [3, 384, 384] vs [3, 1, 1]
    torch::Tensor mean = torch::tensor({0.485f, 0.456f, 0.406f}, device).view({3, 1, 1});
    torch::Tensor stdev = torch::tensor({0.229f, 0.224f, 0.225f}, device).view({3, 1, 1});

    torch::Tensor normalized = (tensor_gpu - mean) / stdev;

    // Add the batch dimension: [3, 384, 384] -> [1, 3, 384, 384]
    torch::Tensor input = normalized.unsqueeze(0);

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Input shape: " << input.sizes() << std::endl;
    // Before normalizing values were in [0, 1]; now they should be roughly in [-2.1, 2.6]
    std::cout << "Min: " << input.min().item<float>()
            << " | Max: " << input.max().item<float>() << std::endl;

    torch::NoGradGuard no_grad;

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);
    std::vector<torch::jit::IValue> inputs_mendicant = inputs;

    // forward returns an IValue (the "box"), so we unpack the tensor inside it
    torch::Tensor logits = Daowa.forward(inputs).toTensor();

    // Raw network output -> probabilities in [0, 1]
    torch::Tensor probs = torch::sigmoid(logits);

    // First RGB and after probs (the Daowa's output mask)
    inputs_mendicant.push_back(probs);
    auto logits_mendicant = Mendicant.forward(inputs_mendicant).toTensor();
    std::cout << "Mendicant Bias output shape: " << logits_mendicant.sizes() << std::endl;

    auto output = torch::softmax(logits_mendicant, 1);

    auto outputdev = output.argmax(1);
    auto value = outputdev.cpu().item<int64_t>();

    for (int i = 0; i < output.sizes()[0]; i++) {
        for (int j = 0; j < output.sizes()[1]; j++) {
            std::cout << outputs[j] << " confidence: "
                    << output[i][j].item<float>() * 100 << "%." << std::endl;
        }
    }

    std::cout << "Mendicant Bias output: " << outputs[value] << std::endl;


    // Tests: Daowa-maad with a real image
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Daowa output shape (It has to be: [1, 1, 384, 384]): " << probs.sizes() << std::endl;
    std::cout << "Logits -> Min: " << logits.min().item<float>()
            << " | Max: " << logits.max().item<float>() << std::endl;
    std::cout << "Probs  -> Min: " << probs.min().item<float>()
            << " | Max: " << probs.max().item<float>() << std::endl;
    // Fraction of pixels the model considers "pet" (prob > 0.5). A real segmentation
    // gives something in between; ~0 or ~1 means it sees nothing / everything
    std::cout << "Pet pixels (prob > 0.5): "
            << (probs > 0.5f).to(torch::kFloat32).mean().item<float>() * 100.0f << "%" << std::endl;

    // ---------------- Visualization ----------------
    // Tensor (GPU) -> cv::Mat (CPU): the reverse of from_blob.
    // [1, 1, 384, 384] bool -> [384, 384] uint8 with 0 (background) / 255 (pet)
    torch::Tensor mask_cpu = (probs > 0.5f).to(torch::kUInt8).mul(255).squeeze().to(torch::kCPU).contiguous();

    // The Mat only WATCHES the tensor's memory (the owner is mask_cpu now)
    cv::Mat mask(384, 384, CV_8UC1, mask_cpu.data_ptr<uint8_t>());
    cv::imwrite(mask_path, mask);

    // Overlay on the original photo (img was overwritten by the resize, so read it again)
    cv::Mat original = cv::imread(image_path);

    // Back to the original size. INTER_NEAREST so the mask stays pure 0/255 (no gray edges)
    cv::Mat mask_full;
    cv::resize(mask, mask_full, original.size(), 0, 0, cv::INTER_NEAREST);

    // Green tint only where the mask is 255
    cv::Mat green(original.size(), original.type(), cv::Scalar(0, 255, 0)); // BGR -> green
    cv::Mat tinted;
    cv::addWeighted(original, 0.55, green, 0.45, 0, tinted);
    cv::Mat overlay = original.clone();
    tinted.copyTo(overlay, mask_full);

    // Outline of the segmentation
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask_full, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    cv::drawContours(overlay, contours, -1, cv::Scalar(0, 255, 0), 6);

    // Original | Overlay, side by side
    cv::Mat side_by_side;
    cv::hconcat(original, overlay, side_by_side);
    cv::imwrite(overlay_path, side_by_side);

    std::cout << "------------------------------------------" << std::endl;
    std::cout << "Saved: " << mask_path << " and " << overlay_path << std::endl;
}

void check_shape (const cv::Mat& image) {
        std::cout << "Rows: " << image.rows
            << "\nColumns: " << image.cols
            << "\nChannels: " << image.channels()
            << std::endl;
}