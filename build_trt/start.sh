#!/bin/sh
docker run -it --gpus all -v ${PWD}:/mnt nvcr.io/nvidia/pytorch:22.12-py3 /mnt/_build.sh
