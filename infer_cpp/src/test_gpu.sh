#!/bin/sh
cd $(dirname $0)
/usr/bin/time ./infer models/model_cuda.pth models/labels.txt true /data/doggo2.jpg 

