#! /bin/bash

#echo "Precision ?  "
#read precision

qsub -q long.q -cwd -V -N essai -b y "python3 essai.py"
