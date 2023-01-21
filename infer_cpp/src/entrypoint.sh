#!/bin/bash
set -e

cd /src/models
python3 resnet.py
cd .. && ./build.sh
cd web && ./build.sh
exec "$@"
