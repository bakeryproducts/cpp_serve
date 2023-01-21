#include "infer.h"


int main(int argc, char **argv) {

  if (argc != 5) {
    std::cerr << "usage: predict <path-to-exported-script-module> <path-to-labels-file> <gpu-flag{true/false} <path-to-image>> \n";
    return -1;
  }

  std::string model_path = argv[1];
  std::string labels_path = argv[2];
  std::string usegpu= argv[3];
  std::string image_path = argv[4];

  torch::DeviceType device_type = (usegpu == "true") ? torch::kCUDA : torch::kCPU;
  torch::Device device(device_type);
  torch::jit::script::Module model = read_model(model_path, device);

  int image_height = 0;
  int image_width = 0;


  // Read labels
  std::vector<std::string> labels;
  std::string label;
  std::ifstream labelsfile (labels_path);
  if (labelsfile.is_open()) {
    while (getline(labelsfile, label)) {
      labels.push_back(label);
    }
    labelsfile.close();
  }

  cv::Mat image = cv::imread(image_path);

  std::string pred, prob;
  tie(pred, prob) = infer(image, image_height, image_width, labels, model, device);

  std::cout << "PREDICTION  : " << pred << std::endl;
  std::cout << "CONFIDENCE  : " << prob << std::endl;

  return 0;
}
