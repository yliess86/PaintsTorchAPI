#!/usr/bin/env bash

nvidia-docker run \
  -it \
  --name PaintstorchAPI \
  -p 8888:8888 \
  -v /home/yliess/Projects:/Projects \
  pytorch/pytorch:latest
