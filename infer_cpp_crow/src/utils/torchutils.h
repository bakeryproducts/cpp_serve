#ifndef TORCHUTILS_H // To make sure you don't declare the function more than once by including the header multiple times.
#define TORCHUTILS_H

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>

#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

torch::jit::script::Module read_model(std::string, torch::Device);
torch::Tensor forward(std::vector<cv::Mat>, torch::jit::script::Module, torch::Device);

#endif
