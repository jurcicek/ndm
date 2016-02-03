# Neural Dialogue Manager

This a proof of concept of an end-to-end neural dialogue manager.


# Running on the Sun Grid engine

For example, this runs an experiments with 5 runs of the same experiment
   
    qsub -V -b y -cwd -pe smp 5 python3 ./ndm.py --runs=5 --threads=1