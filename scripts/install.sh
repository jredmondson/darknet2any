#!/usr/bin/env bash

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$(dirname "${SCRIPTS_DIR}")" && pwd)"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

sudo apt install python3 python3-pip python3-venv

if [ -d $PROJECT_DIR/.venv ]; then
  echo "removing old venv"
  rm -rf $PROJECT_DIR/.venv
fi

cd $PROJECT_DIR

echo "creating new venv"
python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[cpu]"

