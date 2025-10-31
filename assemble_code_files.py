#!/usr/bin/env python3
import sys
import os

def get_input_arguments():
    """Parse command-line arguments or prompt for missing inputs."""
    if len(sys.argv) >= 3:
        return sys.argv[1], sys.argv[2]
    
    input_file = sys.argv[1] if len(sys.argv) >= 2 else input("Enter the path to the list of files: ")
    output_file = input("Enter the path for the output file: ")
    
    return input_file, output_file

def is_valid_text_file(file_path):
    """Check if the file exists and is a valid text file."""
    if not os.path.exists(file_path):
        return False
    
    if not os.path.isfile(file_path):
        return False
    
    # Check for binary content
    try:
        with open(file_path, 'rb') as f:
            data = f.read(1024)
            if b'\0' in data:  # Null bytes indicate binary file
                return False
        
        # Try reading with common encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-16']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read(1)
                return True
            except UnicodeDecodeError:
                continue
        
        return False
    except Exception:
        return False

def get_file_extension(file_path):
    """Extract file extension without the dot."""
    _, ext = os.path.splitext(file_path)
    if ext:
        return ext[1:]
    return 'txt'

def get_file_encoding(file_path):
    """Determine suitable encoding for reading the file."""
    for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-16']:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                f.read(1)
            return encoding
        except UnicodeDecodeError:
            continue
    return 'latin-1'  # Fallback that won't raise UnicodeDecodeError

def process_files(input_list_path, output_file_path):
    """Process files from the list and create assembled output."""
    if not os.path.exists(input_list_path):
        print(f"Error: Input file '{input_list_path}' does not exist.")
        sys.exit(1)
    
    processed_count = 0
    rejected_files = []
    
    try:
        # Read the list of files
        with open(input_list_path, 'r', encoding='utf-8') as list_file:
            file_paths = [line.strip() for line in list_file if line.strip()]
        
        # Open the output file
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            for file_path in file_paths:
                print(f"Processing: {file_path}")
                
                if not os.path.exists(file_path):
                    print(f"  Skipping: File not found: {file_path}")
                    rejected_files.append((file_path, "File not found"))
                    continue
                
                if is_valid_text_file(file_path):
                    ext = get_file_extension(file_path)
                    
                    # Write header and opening code block
                    output_file.write(f"# {file_path}\n")
                    output_file.write(f"```{ext}\n")
                    
                    # Read and write file content
                    try:
                        encoding = get_file_encoding(file_path)
                        with open(file_path, 'r', encoding=encoding) as input_file:
                            output_file.write(input_file.read())
                    except Exception as e:
                        print(f"  Warning: Error reading file: {e}")
                        rejected_files.append((file_path, str(e)))
                        continue
                    
                    # Write closing code block and blank line
                    output_file.write("\n```\n\n")
                    
                    processed_count += 1
                    print(f"  Added to output: {file_path}")
                else:
                    print(f"  Skipping: Not a valid text file: {file_path}")
                    rejected_files.append((file_path, "Not a valid text file"))
        
        return processed_count, rejected_files
    
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

def main():
    # Get input and output file paths
    input_list_path, output_file_path = get_input_arguments()
    
    print(f"Reading file list from: {input_list_path}")
    print(f"Writing assembled output to: {output_file_path}")
    
    # Process the files
    processed_count, rejected_files = process_files(input_list_path, output_file_path)
    
    # Print summary statistics
    print("\nProcessing complete!")
    print(f"Files successfully processed: {processed_count}")
    
    if rejected_files:
        print(f"Files rejected: {len(rejected_files)}")
        print("\nRejected files:")
        for file_path, reason in rejected_files:
            print(f"  - {file_path}: {reason}")
    else:
        print("No files were rejected.")

if __name__ == "__main__":
    main()
