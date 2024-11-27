#!/bin/bash
#SBATCH --partition=gpus         # Partition to run on
#SBATCH --gres=gpu:8             # Request 8 GPU resources
#SBATCH --time=12:00:00          # Request 12 hours of runtime
#SBATCH --mem=32G                # Request 32GB of memory
#SBATCH -J kevin        # Specify a job name
#SBATCH -o kevin-%j.out # Specify an output file
#SBATCH -e kevin-%j.err # Specify an error file

# Activate the virtual environment
source ~/pobax/bin/activate

# Specify the filename
input_file="../runs/runs_ant_ppo_LD.txt"
job_name=$(basename "$input_file" .txt) # Extract the base name of the file without extension

# Read commands from the file and submit each as a separate job
while IFS= read -r command; do
    sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --mem=64G
#SBATCH -J ${job_name}
#SBATCH -o ${job_name}_%j.out
#SBATCH -e ${job_name}_%j.err

# Activate the virtual environment
source ~/pobax/bin/activate

# Run the command
$command
EOT
done < "$input_file"