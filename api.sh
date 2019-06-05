#!/usr/bin/env bash

pip install -r requirements.txt

python3 api.py \
  -d cpu \
  -g '/Projects/PaintsTorchExp/test/generator.pth' \
  -i '/Projects/PaintsTorch/res/model/i2v.pth'
