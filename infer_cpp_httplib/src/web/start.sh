#!/bin/sh
set -e
cd $(dirname $0)
./start_cpu.sh & ./start_gpu.sh
