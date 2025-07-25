#!/usr/bin/env bash

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$(dirname "${SCRIPTS_DIR}")" && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

sudo apt install python3 python3-pip python3-venv

cd $PROJECT_DIR
python3 -m venv .venv
source .venv/bin/activate

pip install -e .[tensorrt]
