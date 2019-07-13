#!/usr/bin/env bash

export PYTHONIOENCODING=UTF-8

pip install -r requirements.txt

python3 api.py \
  -d cuda \
  -g '/Projects/PaintsTorchExp/trained/PaperRS/generator.pth' \
     '/Projects/PaintsTorchExp/trained/CustomSS/generator.pth' \
     '/Projects/PaintsTorchExp/trained/CustomSD/generator.pth' \
  -n 'PaperRS' 'CustomSS' 'CustomSD' \
  -i '/Projects/PaintsTorch/res/model/i2v.pth' \
  -s '/Projects/PaintsTorchStats/data/StudyDataset'
