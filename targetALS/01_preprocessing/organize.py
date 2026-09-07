import os
import re
import shutil

def organize_by_surrogate():
    # Define the target directory
    directory = r'C:\Users\riccardo-s\Documents\CNT\targetALS\exp7\24h\v2'
    
    if not os.path.exists(directory):
        print(f"Directory {directory} not found.")
        return

    # Pattern to find the number 'n' inside '_surrogate_0n_'
    # This captures the digits immediately following 'surrogate_'
    pattern = re.compile(r'_surrogate_(\d+)_')

    files_processed = 0

    for filename in os.listdir(directory):
        # We only want to process .xlsx files
        if filename.endswith(".xlsx"):
            match = pattern.search(filename)
            
            if match:
                # Extract the number (e.g., '01', '02')
                n_value = match.group(1)
                
                # Create the path for the new subfolder (e.g., .../v2/01)
                target_folder = os.path.join(directory, n_value)
                
                # Create folder if it doesn't exist
                if not os.path.exists(target_folder):
                    os.makedirs(target_folder)
                
                # Define source and destination paths
                source_path = os.path.join(directory, filename)
                destination_path = os.path.join(target_folder, filename)
                
                # Move the file
                try:
                    shutil.move(source_path, destination_path)
                    print(f"Moved: {filename} -> Folder {n_value}")
                    files_processed += 1
                except Exception as e:
                    print(f"Error moving {filename}: {e}")

    print(f"\nTask complete. Total files moved: {files_processed}")

if __name__ == "__main__":
    organize_by_surrogate()