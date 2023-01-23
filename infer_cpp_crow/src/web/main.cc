#include "../infer.h"
#include "crow_all.h"
#include "base64.h"
#include <chrono>

using namespace std;

int PORT;

int main(int argc, char **argv)
{

  if (argc != 5)
  {
    std::cerr << "usage: predict <path-to-exported-script-module> <path-to-labels-file> <gpu-flag{true/false}> \n";
    return -1;
  }

  std::string model_path = argv[1];
  std::string labels_path = argv[2];
  std::string usegpu = argv[3];
  PORT = atoi(argv[4]);

  std::string device_str = (usegpu == "true") ? "gpu" : "cpu";
  torch::DeviceType device_type = (usegpu == "true") ? torch::kCUDA : torch::kCPU;
  torch::Device device(device_type);

  torch::jit::script::Module model = read_model(model_path, device);

  // not using resize atm, happens on graph
  int image_height = 0;
  int image_width = 0;

  // Read labels from .txt file
  std::vector<std::string> labels;
  std::string label;
  std::ifstream labelsfile(labels_path);
  if (labelsfile.is_open())
  {
    while (getline(labelsfile, label))
    {
      labels.push_back(label);
    }
    labelsfile.close();
  }

  crow::SimpleApp app;
  CROW_ROUTE(app, "/infer_cpp_crow/single").methods("POST"_method, "GET"_method)
  ([&image_height, &image_width, &labels, &model, &device](const crow::request& req){
    crow::json::wvalue result;
    result["Prediction"] = "";
    result["Confidence"] = "";
    result["Status"] = "Failed";
    std::ostringstream os;

    try {
      auto args = crow::json::load(req.body);

      std::string base64_image = args["image"].s();
      std::string decoded_image = base64_decode(base64_image);
      std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
      cv::Mat image = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);

      std::string pred, prob;
      tie(pred, prob) = infer(image, image_height, image_width, labels, model, device);

      result["Prediction"] = pred;
      result["Confidence"] = prob;
      result["Status"] = "Success";

      os << crow::json::dump(result);
      return crow::response{os.str()};

    } catch (std::exception& e){
      os << crow::json::dump(result);
      return crow::response(os.str());
    }

  });

  CROW_ROUTE(app, "/infer_cpp_crow/bench").methods("POST"_method, "GET"_method)
  ([&image_height, &image_width, &labels, &model, &device](const crow::request& req){
    crow::json::wvalue result;
    result["Prediction"] = "";
    result["Confidence"] = "";
    result["Status"] = "Failed";
    std::ostringstream os;

    try {
      auto args = crow::json::load(req.body);

      std::string base64_image = args["image"].s();
      std::string decoded_image = base64_decode(base64_image);
      std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
      cv::Mat image = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);

      std::string pred, prob;
      tie(pred, prob) = infer(image, image_height, image_width, labels, model, device);

      std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
      int N = 100;
      for(int j=0; j<N; j++){
            tie(pred, prob) = infer(image, image_height, image_width, labels, model, device);
      }
      std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
      std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "[ms]" << std::endl;
      chrono::duration<double> elapsed = end - start;
      auto x = chrono::duration_cast<chrono::nanoseconds>(elapsed);
      std::string avg_time = std::to_string(x.count() / pow(10,9) / N);


      result["Prediction"] = pred;
      result["Confidence"] = prob;
      result["average_time_seconds"] = avg_time;
      result["Status"] = "Success";

      os << crow::json::dump(result);
      return crow::response{os.str()};

    } catch (std::exception& e){
      os << crow::json::dump(result);
      return crow::response(os.str());
    }

  });

  app.port(PORT).run();
  return 0;
}
