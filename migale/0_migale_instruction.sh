#! /bin/bash

echo "nb_step ?"
read nb

for ((i=0; i<nb; i++))
do
	qsub -q long.q -cwd -V -N sp4_$i -b y "python3 sp4_$i.py"
done
