#include "torchutils.h"


torch::Tensor __convert_image_to_tensor(cv::Mat image) {
  int n_channels = image.channels();
  int height = image.rows;
  int width = image.cols;
  int image_type = image.type();

  assert((image_type % 8 == 0) || ((image_type - 5) % 8 == 0) || ((image_type - 6) % 8 == 0));

  std::vector<int64_t> dims = {height, width, n_channels};
  std::vector<int64_t> permute_dims = {2, 0, 1};

  torch::Tensor image_as_tensor;
  if (image_type % 8 == 0) {
    torch::TensorOptions options(torch::kUInt8);
    image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
  } else if ((image_type - 5) % 8 == 0) {
    torch::TensorOptions options(torch::kFloat32);
    image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
  } else if ((image_type - 6) % 8 == 0) {
    torch::TensorOptions options(torch::kFloat64);
    image_as_tensor = torch::from_blob(image.data, torch::IntList(dims), options).clone();
  }

  image_as_tensor = image_as_tensor.permute(torch::IntList(permute_dims));
  image_as_tensor = image_as_tensor.toType(torch::kFloat32);
  
  return image_as_tensor;
}

torch::Tensor __predict(torch::jit::script::Module model, torch::Tensor tensor) {

  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(tensor);

  torch::NoGradGuard no_grad;
  torch::Tensor output = model.forward(inputs).toTensor();

  torch::DeviceType cpu_device_type = torch::kCPU;
  torch::Device cpu_device(cpu_device_type);
  output = output.to(cpu_device);

  return output;
}


torch::jit::script::Module read_model(std::string model_path, torch::Device device) {
  torch::jit::script::Module model = torch::jit::load(model_path);
  model.to(device);
  return model;
}


torch::Tensor forward(std::vector<cv::Mat> images, torch::jit::script::Module model, torch::Device device) {

  torch::Tensor tensor = __convert_image_to_tensor(images[0]);
  tensor = tensor.to(device);
  torch::Tensor output = __predict(model, tensor);
  return output;
}
