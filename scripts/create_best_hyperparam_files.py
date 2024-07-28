from pathlib import Path
from definitions import ROOT_DIR

def create_best_files(directory):
    # Convert string directory to a Path object for better path handling
    dir_path = directory
    for file_path in dir_path.glob('*.py'):  # Glob to filter only Python files
        filename = file_path.name
        if filename.endswith(".py"):
            new_filename = filename.replace(".py", "_best.py")
            new_file_path = file_path.parent / new_filename  # Use Path operations for new file path
            new_file_path.touch()  # Create the new file
            print(f"Created {new_filename}")

# Example usage
if __name__ == '__main__':
    directory_path = Path(ROOT_DIR, 'scripts', 'hyperparams')  # Replace with the actual directory path
    create_best_files(directory_path)

