from pathlib import Path
from definitions import ROOT_DIR

def rename_horizon_files(directory):
    dir_path = Path(directory)  # Convert string directory to a Path object for better path handling
    for file_path in dir_path.glob('*.py'):  # Glob to filter only Python files
        filename = file_path.name
        if 'horizon5' in filename:  # Check if 'horizon5' is in the filename
            new_filename = filename.replace('horizon5', 'horizon7')
            new_file_path = file_path.parent / new_filename  # Use Path operations for new file path
            file_path.rename(new_file_path)  # Rename the file
            print(f"Renamed {filename} to {new_filename}")

# Example usage
if __name__ == '__main__':
    directory_path = Path(ROOT_DIR, 'scripts', 'hyperparams', 'rnn_approximator_horizon7_hidden_size')  # Replace with the actual directory path
    rename_horizon_files(directory_path)