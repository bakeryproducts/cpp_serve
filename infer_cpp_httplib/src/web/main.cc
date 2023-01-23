#include "../infer.h"
#include "httplib.h"
#include "base64.h"
#include <chrono>

using namespace httplib;
using namespace std;

string HOST = "0.0.0.0";
int PORT;

string to_json(string pred, string prob)
{
  // looks ugly TODO: do that normaly
  string json = "{\"" + string("Prediction") + "\":";
  json += "\"" + pred + "\"";
  json += ",";
  json += "\"" + string("Confidence") + "\":";
  json += "\"" + prob + "\"";
  json += "}";
  return json;
}

int main(int argc, char **argv)
{

  if (argc != 5)
  {
    std::cerr << "usage: predict <path-to-exported-script-module> <path-to-labels-file> <gpu-flag{true/false}> <port int> \n";
    return -1;
  }

  std::string model_path = argv[1];
  std::string labels_path = argv[2];
  std::string usegpu = argv[3];
  PORT = atoi(argv[4]);

  string device_str = (usegpu == "true") ? "gpu" : "cpu";
  torch::DeviceType device_type = (usegpu == "true") ? torch::kCUDA : torch::kCPU;
  torch::Device device(device_type);

  // read presaved python JIT model, put in on device
  torch::jit::script::Module model = read_model(model_path, device);

  // not using resize atm
  int image_height = 0;
  int image_width = 0;

  // Read labels from txt file
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

  // cpp-httlib server, looking just for post reqs
  Server svr;

  // dynamic url, runtime choice
  string post_url = "/infer_cpp_httplib/" + device_str + "/single";
  svr.Post(post_url, [&](const Request &req, Response &res)
           {
             //std::cout << "DEBUG look at ya, posting cringe!" << std::endl;
             auto image_file = req.get_file_value("image_file");

             // i cant figure out how to do send-recieve POST jsons without base64 encoding, leaving it as is.
             std::string base64_image = image_file.content;
             std::string decoded_image = base64_decode(base64_image);
             std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
             cv::Mat image = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);

             std::string pred, prob;
             tie(pred, prob) = infer(image, image_height, image_width, labels, model, device);

             string json = to_json(pred, prob);
             res.set_content(json, "appliation/json");
             res.status = 200;
           });

  // dynamic url for benchmarking option
  string post_url_bench = "/infer_cpp_httplib/" + device_str + "/bench";
  svr.Post(post_url_bench, [&](const Request &req, Response &res)
           {
             //std::cout << " look at ya, posting cringe!" << std::endl;
             auto image_file = req.get_file_value("image_file");

             std::string base64_image = image_file.content;
             std::string decoded_image = base64_decode(base64_image);
             std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
             cv::Mat image = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);

             std::string pred, prob;
             tie(pred, prob) = infer(image, image_height, image_width, labels, model, device);

             // timing 100 runs, can send N option with post probably or something. cannot figure out dynamic request for url
             std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
             int N = 100;
             for (int j = 0; j < N; j++)
             {
               tie(pred, prob) = infer(image, image_height, image_width, labels, model, device);
             }
             std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
             std::cout << "Time difference = " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "[ms]" << std::endl;
             chrono::duration<double> elapsed = end - start;
             auto x = chrono::duration_cast<chrono::nanoseconds>(elapsed);
             string avg_time = to_string(x.count() / pow(10, 9) / N);

             string json = to_json(pred, prob);
             // well here we go again. TODO: fix this
             json.pop_back();
             json += ",\"" + string("average_time_seconds") + "\":";
             json += "\"" + avg_time + "\",";
             json += "\"" + string("num_runs") + "\":";
             json += "\"" + to_string(N) + "\"}";
             res.set_content(json, "appliation/json");
             res.status = 200;
           });

  svr.listen(HOST, PORT);

  return 0;
}
