#!/usr/bin/env python3
import argparse
from pathlib import Path

def combine_run_files(input_files, output_file):
    output_path = Path(output_file)
    with output_path.open('w') as outfile:
        for file in input_files:
            in_path = Path(file)
            if not in_path.is_file():
                print(f"Warning: {file} does not exist or is not a file; skipping.")
                continue
            with in_path.open('r') as infile:
                for line in infile:
                    # Optionally, you can skip blank lines or comments
                    if line.strip() == "" or line.strip().startswith("#"):
                        continue
                    outfile.write(line)
    print(f"Combined run file written to: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Combine multiple run files into a single file."
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="Paths to input run files (each containing experiment commands)."
    )
    parser.add_argument(
        "--output", "-o",
        default="combined_runs.txt",
        help="Path to the output run file. Default is 'combined_runs.txt'."
    )
    args = parser.parse_args()

    combine_run_files(args.input_files, args.output)

if __name__ == '__main__':
    main()
