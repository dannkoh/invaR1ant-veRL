#$ -l a100=2
#$ -pe smp.pe 24
#$ -o output.log
#$ -e error.log
#$ -m bea
#$ -M daniel.koh@student.manchester.ac.uk

# Load necessary modules
module load libs/cuda/12.4.1
module load compilers/gcc/13.3.0

# Set environment variables
export LD_PRELOAD=/opt/apps/compilers/gcc/13.3.0/lib64/libstdc++.so.6
export HF_HOME=$TMPDIR

# Activate virtual environment
cd ConStruct-veRL/
source .venv/bin/activate

./scripts/qwen2.5-3b-base-ppo.sh