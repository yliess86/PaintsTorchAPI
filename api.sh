#!/usr/bin/env bash

export PYTHONIOENCODING=UTF-8

pip install -r requirements.txt

python3 api.py \
  -d cuda \
  -g '/Projects/PaintsTorchExp/trained/prs/generator.pth' \
     '/Projects/PaintsTorchExp/trained/pss/generator.pth' \
     '/Projects/PaintsTorchExp/trained/psd/generator.pth' \
     '/Projects/PaintsTorchExp/trained/csd/generator.pth' \
     '/Projects/PaintsTorchExp/custom_strokes_double/model/generator.pth' \
  -n 'PRS' 'PSS' 'PSD' 'CSD' 'FTM' \
  -i '/Projects/PaintsTorch/res/model/i2v.pth'
