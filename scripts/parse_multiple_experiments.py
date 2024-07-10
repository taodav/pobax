import subprocess
from pathlib import Path
from definitions import ROOT_DIR

# Define the root directory containing the results
RESULTS_DIR = Path(ROOT_DIR) / "results"

# Define the scripts to run
parse_script = "python parse_batch_experiments.py"
best_parse_script = "python best_parse_batch_experiments.py"
best_hyperparams_script = "python best_hyperparams_per_env.py"

# Iterate over each subdirectory in the results directory
for subdir in RESULTS_DIR.iterdir():
    if subdir.is_dir():
        print(f"Processing directory: {subdir}")

        try:
            # Determine which parse script to use
            if subdir.name.endswith("best"):
                parse_command = f"{best_parse_script} {subdir}"
            else:
                parse_command = f"{parse_script} {subdir}"

            print(f"Running: {parse_command}")
            subprocess.run(parse_command, shell=True, check=True)

            if not subdir.name.endswith("best"):
                # Construct the path to the parsed_hparam_scores.pkl file
                parsed_pkl_path = subdir / "parsed_hparam_scores.pkl"

                # Check if the parsed_pkl_path exists before proceeding
                if not parsed_pkl_path.exists():
                    print(f"File {parsed_pkl_path} not found. Skipping best hyperparameters extraction.")
                else:
                    # Run the best_hyperparams_per_env.py script
                    best_hyperparams_command = f"{best_hyperparams_script} {parsed_pkl_path}"
                    print(f"Running: {best_hyperparams_command}")
                    subprocess.run(best_hyperparams_command, shell=True, check=True)

        except subprocess.CalledProcessError as e:
            print(f"Error processing directory {subdir}: {e}")
            continue

print("All processing complete.")



