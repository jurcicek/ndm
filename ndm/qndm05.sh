#!/usr/bin/env bash

qsub -V -b y -cwd -pe smp 5 -o ./log -e ./log -N ndm.py python3 ./ndm.py --runs=5 --threads=1 $@


