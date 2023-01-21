#!/bin/sh
cd $(dirname $0)
./predict ../models/model_cuda.pth ../models/labels.txt true
