#!/usr/bin/env bash

rm `ls -t ./log/*.hpc| awk 'NR>1'`

python2 /usr/local/lib/python2.7/dist-packages/tensorflow/tensorboard/tensorboard.py --logdir ./log