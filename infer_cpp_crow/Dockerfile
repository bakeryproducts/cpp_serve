FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
         g++ \
         make \
         cmake \
         time \
         wget \
         unzip \
         python3 \
         python3-pip \
         libopencv-dev \
         libboost-all-dev

RUN pip3 install torchvision torch

RUN mkdir /cache && cd /cache && wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip
RUN cd /cache && unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip
RUN mkdir /src && mv /cache/libtorch /src/libtorch
COPY infer_cpp_crow/src /src
ENTRYPOINT ["/src/entrypoint.sh"]
CMD ["/src/web/start.sh"]
