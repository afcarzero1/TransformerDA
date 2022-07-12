#!/bin/bash
#SBATCH -p m100_usr_prod
#SBATCH --time 14:16:00     # format: HH:MM:SS
#SBATCH -N 1                # 1 node
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1 # 8 tasks out of 128
#SBATCH --gres=gpu:4     # 1 gpus per node out of 4
#SBATCH --mem=100000          # memory per node out of 246000MB
#SBATCH --job-name=test
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mirco.planamente@polito.it


source1=$1 #D1
source2=$2 #D1
target=$3 #D1
file=$4
radius=$5

module load profile/deeplrn
module load autoload /epic-kitchens

chmod +x /m100/home/userexternal/abottin1/Mirco_ActivityRecognition_DA/scripts_mirco_Journal_RNA/MultiDG/$file
srun /m100/home/userexternal/abottin1/Mirco_ActivityRecognition_DA/scripts_mirco_Journal_RNA/MultiDG/$file $source1 $source2 $target $radius


