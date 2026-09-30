#pragma once

#include <torch/script.h>
#include <string>

c10::Device get_device();
bool load_model(const std::string& path, torch::jit::script::Module& output);