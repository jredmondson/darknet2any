#!/usr/bin/env bash

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$(dirname "${SCRIPTS_DIR}")" && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

sudo apt install python3 python3-pip python3-venv
sudo sudo apt install -y migraphx

cd $PROJECT_DIR
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements-amd.txt
pip3 install onnxruntime-rocm -f https://repo.radeon.com/rocm/manylinux/rocm-rel-6.4.2/
python setup.py install

