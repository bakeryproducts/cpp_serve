#!/bin/sh
cd $(dirname $0)
./infer models/model_cpu.pth models/labels.txt false /data/doggo2.jpg 

