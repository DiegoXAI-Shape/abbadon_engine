#include <iostream>
#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>
#include "abbadon/check.hpp"

void check_cuda () {
    // Check CUDA
    std::cout << "Is CUDA available? Response: " << torch::cuda::is_available() << std::endl;
    auto* ptr = at::cuda::getDeviceProperties(0);
    std::cout << "GPU name: " << ptr->name << std::endl;
}