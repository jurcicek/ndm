#!/usr/bin/env bash

qsub -V -b y -cwd -pe smp 10 -o ./log -e ./log -N ndm.py python3 ./ndm.py --runs=10 --threads=1 $@


