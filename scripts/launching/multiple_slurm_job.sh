#!/bin/bash
#SBATCH --partition=gpus         # Partition to run on
#SBATCH --gres=gpu:8             # Request 8 GPU resources
#SBATCH --time=12:00:00          # Request 12 hours of runtime
#SBATCH --mem=32G                # Request 32GB of memory
#SBATCH -J MyParallelJobs        # Specify a job name
#SBATCH -o MyParallelJobs-%j.out # Specify an output file
#SBATCH -e MyParallelJobs-%j.err # Specify an error file

# Activate the virtual environment
source ~/pobax/bin/activate
cd ..
python write_multiple_jobs.py 

# Iterate over each .txt file in the scripts/runs directory
for file in runs/*.txt; do
    echo "Processing $file"
    # Extract the base name of the file without the extension to use as the job name
    job_name=$(basename "$file" .txt)
    
    # Read commands from the file and submit each as a separate job
    while IFS= read -r command; do
        sbatch <<EOT
#!/bin/bash
#SBATCH --partition=gpus
#SBATCH --gres=gpu:8
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH -J $job_name
#SBATCH -o ${job_name}-%j.out
#SBATCH -e ${job_name}-%j.err

# Activate the virtual environment
source ~/pobax/bin/activate

# Run the command
$command
EOT
    done < "$file"
done

