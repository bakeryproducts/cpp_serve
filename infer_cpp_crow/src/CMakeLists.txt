cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(cnn-classification)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(infer main.cc infer.cc utils/opencvutils.cc utils/torchutils.cc)

target_link_libraries(infer "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")

message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
message("OpenCV_LIBS: " ${OpenCV_LIBS})

set_property(TARGET infer PROPERTY CXX_STANDARD 14)
set_property(TARGET infer PROPERTY OUTPUT_NAME infer)

