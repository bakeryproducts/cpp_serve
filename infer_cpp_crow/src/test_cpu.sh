#!/bin/sh
cd $(dirname $0)
/usr/bin/time ./infer models/model_cpu.pth models/labels.txt false /data/doggo2.jpg 

