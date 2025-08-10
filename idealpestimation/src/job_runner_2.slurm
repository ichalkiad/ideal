#!/bin/bash
#SBATCH --job-name=expIII_mle
#SBATCH --output=%x_%A_%a.out # create output files indexed with the job array task id
#SBATCH --partition=prepost  #cpu_p1
#SBATCH --time=02:00:00   # DMLE, ICM-D: 2h, CA: 30min, ICM-P: 18h (max)
#SBATCH --qos=qos_cpu-t3   #qos_cpu-dev
#SBATCH --hint=nomultithread 
#SBATCH --nodes=1          
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1  ############################  2 for CA, 3 for ICM-P on cpu_p1, 1 on prepost
#SBATCH --array=684,685,686,687,688,689,1820,1821,1822,1823,1824,1825,1826,2084,2085,2086,2087,2088,2089,2090,2567,2568,2569,2570,2571,2572,2573,2615,2616,2617,2618,2619,2620,2701,2702,2703,2704,2705,2706,2707,3137,3138,3139,3140,3141,3142,3143,3498,3499,3500,3501,3502,3503,3504,3680,3681,3682,3683,3684,3685,3686,3717,3718,3719,3720,3721,3722,3723,4378,4379,4380,4381,4382,4383,4384,4484,4485,4486,4487,4488,4489,4490,4491,4492,4493,4494,4495,4496,4497,4541,4542,4543,4544,4545,4546,4547,4633,4634,4635,4636,4637,4638,4639   # expI ca, icm-p  # 0-1649 expI icm-d, mle   , expIII: ca, icmp 0-29, mle: 0-4659,
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ioannis.chalkiadakis@cnrs.fr

# grep -H "Convergence: True"  expIII_icmd_* | bash -c 'comm -23 <(seq "$1" "$2") <(sed -n "s/.*_\([0-9]\+\)\.out:.*/\1/p" | sort -n | uniq) | paste -sd, - > output.txt' _ 0 4659




cd ${SLURM_SUBMIT_DIR}

export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
module purge
module load singularity

#index in the csv file of each parameter vector
# slurm_array_task_id=int(os.environ["SLURM_ARRAY_TASK_ID"])

set -x


# time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/icm_annealing_posteriorpower_slurm.py"
# time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/icm_annealing_data_slurm.py"
time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/mle_slurm.py"
# time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/ca_slurm.py"

# time singularity exec --bind $WORK:/linkhome/rech/genpuz01/umi36fq,$WORK/ideal/:/app/repo $SINGULARITY_ALLOWED_DIR/ideal_symblink.sif bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate ideal && python -u /app/repo/idealpestimation/src/aggregate_expI.py"
                                                                                                           
