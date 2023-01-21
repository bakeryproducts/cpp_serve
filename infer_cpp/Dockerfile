FROM debian:stable as builder
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
         python3 \
         python3-pip

RUN pip3 install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install numpy opencv-python
RUN apt-get install -y \
         libgl1 \ 
         libglib2.0-0 \
         wget \
         unzip

RUN mkdir /cache && cd /cache && wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
# ADD cache /cache
RUN cd /cache && unzip libtorch-cxx11-abi-shared-with-deps-1.13.1+cpu.zip

ADD src_py /src
ADD data /data
RUN cd /src && python3 resnet.py


FROM debian:stable
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
         g++ \
         make \
         cmake \
         libopencv-dev \
         libboost-all-dev

ADD data /data
ADD src /src
COPY --from=builder /src/model_cpu.pth /src/models/model_cpu.pth
COPY --from=builder /src/labels.txt /src/models/labels.txt
COPY --from=builder /cache/libtorch /src/libtorch

RUN cd /src && ./build.sh && ./test.sh
RUN cd /src/web/ && ./build.sh 
