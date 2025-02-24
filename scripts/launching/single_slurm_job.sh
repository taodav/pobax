#!/bin/bash
#SBATCH --partition=gpus         # Partition to run on
#SBATCH --gres=gpu:1             # Request 8 GPU resources
#SBATCH --exclude=gpu1601,gpu1602,gpu1603,gpu1604,gpu1605,gpu1701,gpu1702,gpu1703,gpu1704,gpu1705,gpu1706,gpu1707,gpu1708,gpu2201,gpu2301,gpu1801,gpu1802,gpu2002
#SBATCH --time=12:00:00          # Request 12 hours of runtime
#SBATCH --mem=32G                # Request 32GB of memory
#SBATCH -J kevin        # Specify a job name
#SBATCH -o kevin-%j.out # Specify an output file
#SBATCH -e kevin-%j.err # Specify an error file

# Activate the virtual environment
source ~/pobax/bin/activate

# Specify the filename
input_file="../runs/runs_hopper_memoryless_ppo_no_frame_madrona.txt"
job_name=$(basename "$input_file" .txt) # Extract the base name of the file without extension

# Read commands from the file and submit each as a separate job
while IFS= read -r command; do
    sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu1601,gpu1602,gpu1603,gpu1604,gpu1605,gpu1701,gpu1702,gpu1703,gpu1704,gpu1705,gpu1706,gpu1707,gpu1708,gpu2201,gpu2301,gpu1801,gpu1802,gpu1905,gpu1906
#SBATCH --time=72:00:00
#SBATCH --mem=48G
#SBATCH -J ${job_name}
#SBATCH -o ${job_name}_%j.out
#SBATCH -e ${job_name}_%j.err

# Activate the virtual environment
source ~/pobax/bin/activate

# Run the command
$command
EOT
done < "$input_file"
