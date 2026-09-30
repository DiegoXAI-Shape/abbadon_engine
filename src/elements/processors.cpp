#include <torch/script.h>
#include <string>
#include <iostream>
#include "abbadon/processors.hpp"

c10::Device get_device() {
    c10::Device device(torch::kCUDA, 0);

    return device;
}

bool load_model(const std::string& path, torch::jit::script::Module& output) {

    auto device = get_device();

    try {

        output = torch::jit::load(path, device);
        return true;
        
    } catch (const c10::Error& e) {

        std::cerr << "Woops... an error has ocurred: " << e.what() << std::endl;
        return false;
    
    }
}