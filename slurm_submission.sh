#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4                    
#SBATCH -p cascade.p
#SBATCH --mem 32G         
#SBATCH --job-name="GNN"
#SBATCH -t 24:00:00
#SBATCH --output=job.o%j
#SBATCH --error=job.e%j

module purge

source /etc/profile.d/modules.sh
CONDA_PREFIX="/hits/fast/mbm/treydewk/conda"
export CONDA_PREFIX
eval "$($CONDA_PREFIX/bin/conda shell.bash hook)"

echo "Script start: `date`"

conda activate conda_tflow
cd /hits/fast/mbm/treydewk/documentation/docs/
python ./train_.py

echo "Started: `date`"
for job in `jobs -p`
do
    wait $job
    echo "$job done"
done
echo "jobs done: `date`"