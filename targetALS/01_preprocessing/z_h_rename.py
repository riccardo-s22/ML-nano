import pandas as pd
import re
import os
import glob
import argparse

def process_folder(folder_path):
    # Ensure the folder path exists
    if not os.path.isdir(folder_path):
        print(f"Error: The directory '{folder_path}' does not exist.")
        return

    # Find all files matching the pattern z_*h.csv
    search_pattern = os.path.join(folder_path, "z_*h.csv")
    files = glob.glob(search_pattern)
    
    if not files:
        print(f"No files matching 'z_*h.csv' found in: {folder_path}")
        return

    print(f"Found {len(files)} files. Starting transformation...")

    for file_path in files:
        filename = os.path.basename(file_path)
        
        # Extract the suffix (e.g., '0h', '24h') from z_(SUFFIX).csv
        match = re.search(r'z_(.*)\.csv', filename)
        if not match:
            continue
            
        suffix = match.group(1)
        
        try:
            # Load the CSV
            df = pd.read_csv(file_path)
            
            # Rename columns that start with 'dim'
            df.columns = [f"{col}_{suffix}" if col.startswith('dim') else col for col in df.columns]
            
            # Define output filename and path
            output_filename = filename.replace('.csv', '_renamed.csv')
            output_path = os.path.join(folder_path, output_filename)
            
            # Save the new version
            df.to_csv(output_path, index=False)
            print(f"  [DONE] {filename} -> {output_filename}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {filename}: {e}")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Rename 'dim' headers in z_*h.csv files based on filename suffix.")
    parser.add_argument("folder", type=str, help="The path to the folder containing the CSV files.")
    
    args = parser.parse_args()
    
    # Run the process
    process_folder(args.folder)