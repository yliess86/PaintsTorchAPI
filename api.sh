#!/usr/bin/env bash

export PYTHONIOENCODING=UTF-8

pip install -r requirements.txt

python3 api.py \
  -d cuda \
  -g '/Projects/PaintsTorchExp/paper_random_simpler/generator.pth' \
     '/Projects/PaintsTorchExp/paper_strokes_simpler/generator.pth' \
     '/Projects/PaintsTorchExp/paper_strokes_double/generator.pth' \
     '/Projects/PaintsTorchExp/custom_strokes_double/generator.pth' \
  -n 'PRS' 'PSS' 'PSD' 'CSD' \
  -i '/Projects/PaintsTorch/res/model/i2v.pth'
