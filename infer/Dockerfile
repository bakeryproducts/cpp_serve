FROM nvcr.io/nvidia/pytorch:22.12-py3
# nvcr one has torchtrt inside

#FROM nvidia/cuda:11.3.0-runtime-ubuntu20.04
#RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
#        python3 \
#        python3-pip
# RUN pip3 install torchvision torch --no-index --find-links /infer/whl

ADD infer/requirements.txt /infer/requirements.txt
WORKDIR /infer
RUN pip3 install -r requirements.txt

ADD infer /infer
