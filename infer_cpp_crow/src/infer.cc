#include "infer.h"

std::tuple<std::string, std::string> infer(

    cv::Mat image,
    int image_height, int image_width,
    std::vector<std::string> labels,
    torch::jit::script::Module model,
    torch::Device device)
{

  std::string pred = "";
  std::string prob = "0.0";

  if (image.empty())
  {
    std::cout << "ERROR: Cannot read image!" << std::endl;
  }
  else
  {
    image = preprocess(image, image_height, image_width);
    torch::Tensor output = forward({
                                       image,
                                   },
                                   model, device); // [CLASS_ID, PROB], [234, .832]
    int class_id;
    float prob_float;
    class_id = output[0].item<int>();
    prob_float = output[1].item<float>();
    pred = labels[class_id];
    prob = std::to_string(prob_float);
  }

  return std::make_tuple(pred, prob);
}
