#!/usr/bin/env bash

export PYTHONIOENCODING=UTF-8

pip install -r requirements.txt

python3 api.py \
  -d cpu \
  -g '/Projects/PaintsTorchExp/test/generator.pth' \
  -i '/Projects/PaintsTorch/res/model/i2v.pth'
