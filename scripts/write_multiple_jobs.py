import os
import subprocess
from pathlib import Path
from definitions import ROOT_DIR

# Directory containing the files
run_dir = Path(ROOT_DIR, 'scripts', 'runs')
hyperparams_dir = Path(ROOT_DIR, 'scripts', 'hyperparams')

# Base command to execute
base_command = "python write_jobs.py"

# Delete all files in the runs directory
for file_path in run_dir.iterdir():
    if file_path.is_file():
        file_path.unlink()  # Delete the file

# Iterate over each file in the hyperparams directory
for file_path in hyperparams_dir.iterdir():
    # Check if it is a file (not a directory)
    if file_path.is_file():
        # Construct and execute the command
        command = f"{base_command} {file_path}"
        subprocess.run(command, shell=True)


