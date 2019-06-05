#!/usr/bin/env bash

nvidia-docker run \
  -it \
  --name PaintstochAPI \
  -v /home/yliess/Projects:/Projects \
  pytorch/pytorch:latest
