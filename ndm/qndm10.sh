#!/usr/bin/env bash

qsub -V -b y -cwd -pe smp 10 -o ./log -e ./log -l mem_free=4G,h_vmem=8G -p -50 -q "`qselect | sort | egrep -v 'pandora|hyperion|orion|andromeda|lucifer|cosmos' | tr '\n' ',' | sed s/\,$//`" -N ndm.py python3 ./ndm.py --runs=10 --threads=1 $@


