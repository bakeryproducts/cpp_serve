# FROM debian:stable as builder
# RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
#          python3 \
#          python3-pip

# RUN pip3 install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
# RUN pip3 install numpy opencv-python
# RUN apt-get install -y \
#          libgl1 \ 
#          libglib2.0-0 \
#          wget \
#          unzip

#RUN mkdir /cache && cd /cache && wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
# ADD cache /cache
#RUN cd /cache && unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# ADD src_py /src
# ADD data /data
# RUN cd /src && python3 resnet.py


# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
#FROM debian:stable
# FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04
# FROM nvidia/cuda:11.7.0-devel-ubuntu18.04
# FROM nvidia/cuda:11.7.0-cudnn8-runtime-ubuntu18.04
# FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu18.04
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04
# FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04
# FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
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

# RUN mkdir /cache && cd /cache && wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
RUN mkdir /cache && cd /cache && wget https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip
# RUN mkdir /cache && cd /cache && wget https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.10.1%2Bcu111.zip
# RUN mkdir /cache && cd /cache && wget https://download.pytorch.org/libtorch/cu110/libtorch-cxx11-abi-shared-with-deps-1.7.1%2Bcu110.zip
RUN cd /cache && unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cu117.zip
# RUN cd /cache && unzip libtorch-cxx11-abi-shared-with-deps-1.7.1+cu110.zip
ADD data /data
RUN mkdir /src && mv /cache/libtorch /src/libtorch
COPY src /src
# COPY --from=builder /src/model_cpu.pth /src/models/model_cpu.pth
# COPY --from=builder /src/labels.txt /src/models/labels.txt
# COPY --from=builder /cache/libtorch /src/libtorch

#RUN cd /src && ./build.sh && ./test.sh
#RUN cd /src/web/ && ./build.sh 
# ENTRYPOINT ["/bin/bash", "/src/entrypoint.sh"]
# ENTRYPOINT ["/bin/bash", "/src/web/tmp.sh", "/bin/bash", "-c"]
ENTRYPOINT ["/src/entrypoint.sh"]
CMD ["/src/web/start_gpu.sh"]
