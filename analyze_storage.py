
import os

def analyze_csv_files(file_paths):
    total_non_zero_frames = 0
    file_counts = {}
    
    print("Starting analysis...")

    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        non_zero_count = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if not lines:
                    print(f"File is empty: {file_path}")
                    continue

                header = lines[0].strip().split(',')
                displacement_col_idx = len(header) - 1
                
                for line in lines[1:]:
                    try:
                        parts = line.strip().split(',')
                        if len(parts) > displacement_col_idx:
                            displacement_val = float(parts[displacement_col_idx])
                            if displacement_val != 0.0:
                                non_zero_count += 1
                    except (ValueError, IndexError):
                        continue
            
            file_counts[os.path.basename(file_path)] = non_zero_count
            total_non_zero_frames += non_zero_count
            print(f"Processed '{os.path.basename(file_path)}': Found {non_zero_count} frames with motion.")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    if not file_counts:
        print("No files were processed successfully.")
        return

    average_frames = total_non_zero_frames / len(file_counts)
    
    print("\n--- Analysis Summary ---")
    for filename, count in file_counts.items():
        print(f"- {filename}: {count} frames")
        
    print(f"\nTotal frames with motion across {len(file_counts)} files: {total_non_zero_frames}")
    print(f"Average frames with motion per file: {average_frames:.2f}")

files_to_check = [
    "D:\\Lift_project\\lifts\\result\\1.csv",
    "D:\\Lift_project\\lifts\\result\\11.csv",
    "D:\\Lift_project\\lifts\\result\\21.csv",
    "D:\\Lift_project\\lifts\\result\\31-1.csv",
    "D:\\Lift_project\\lifts\\result\\41.csv"
]

analyze_csv_files(files_to_check)
