#!/bin/sh
cd $(dirname $0)
./predict ../models/model_cpu.pth ../models/labels.txt false
