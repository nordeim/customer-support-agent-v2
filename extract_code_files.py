#!/usr/bin/env python3
import os
import sys
import re

def get_input_arguments():
    """Parse command-line arguments or prompt for missing input."""
    if len(sys.argv) >= 2:
        return sys.argv[1]
    input_file = input("Enter the path to the compacted input file: ")
    return input_file

def is_readable_text_file(file_path):
    """Check if the file exists and is a readable text file."""
    if not os.path.exists(file_path):
        return False, "File does not exist"
    if not os.path.isfile(file_path):
        return False, "Not a regular file"
    try:
        with open(file_path, 'rb') as f:
            data = f.read(1024)
            if b'\0' in data:
                return False, "File appears to be binary"
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'utf-16']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    f.read()
                return True, encoding
            except UnicodeDecodeError:
                continue
        return False, "Could not determine text encoding"
    except Exception as e:
        return False, str(e)

def normalize_path(path_str):
    """Normalize path by handling quotes and escapes."""
    path_str = path_str.strip()
    if (path_str.startswith("'") and path_str.endswith("'")) or \
       (path_str.startswith("`") and path_str.endswith("`")):
        path_str = path_str[1:-1]
    path_str = path_str.replace("\\ ", " ")
    return path_str.strip()

def ensure_directory_exists(file_path):
    """Create directory for the file if it doesn't exist."""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        try:
            os.makedirs(directory)
            print(f"  Created directory: {directory}")
            return True
        except Exception as e:
            print(f"  Error creating directory {directory}: {e}")
            return False
    return True

def extract_files(input_file_path, encoding):
    """
    Extract files from bundled code blocks with formats like:

    ```tsx
    // app/page.tsx
    import { Button } from '@/components/common/Button'
    import Link from 'next/link'
    ... // other lines 
    ```

    or for Python:

    ```python
    # File: path/to/file.py
    <file contents>
    ```
    
    This script supports various file types including .js, .ts, and .tsx.
    The first line within the code fence should be a marker comment with the file path.
    It accepts either a "# File:" marker or a "//" style marker.
    """
    blocks_found = 0
    successful_extractions = 0
    rejected_blocks = []
    overwrites = []

    # Accept code fence markers for various languages.
    code_fence_pattern = re.compile(r"^```(?:python|py|ts|tsx|js|jsx|json|prisma|css|mjs|typescript|sql)\s*$")
    # Support marker lines that start with "# File:" or with "//"
    file_marker_pattern = re.compile(r"^(?:#\s*File:\s*|//\s*)(.+)$")
    closing_fence_pattern = re.compile(r"^```\s*$")
    created_files = set()

    try:
        with open(input_file_path, 'r', encoding=encoding) as f:
            lines = f.readlines()

        i = 0
        n = len(lines)
        while i < n:
            line = lines[i].rstrip('\r\n')
            # Look for an opening code fence that matches one of the allowed languages.
            if code_fence_pattern.match(line):
                blocks_found += 1
                # Must have at least one more line for the file marker
                if i+1 >= n:
                    rejected_blocks.append(("<unknown>", "Missing file marker after code fence"))
                    i += 1
                    continue
                marker_line = lines[i+1].rstrip('\r\n')
                marker_match = file_marker_pattern.match(marker_line)
                if not marker_match:
                    rejected_blocks.append(("<unknown>", "Expected a file marker comment after code fence"))
                    i += 1
                    continue
                filename = normalize_path(marker_match.group(1))
                content_lines = [marker_line]  # Include the marker line as the first line of the file

                # Collect content lines until the closing fence is encountered.
                j = i+2
                found_closing = False
                while j < n:
                    next_line = lines[j].rstrip('\r\n')
                    if closing_fence_pattern.match(next_line):
                        found_closing = True
                        break
                    content_lines.append(next_line)
                    j += 1
                if not found_closing:
                    rejected_blocks.append((filename, "Missing closing code fence"))
                    i = j
                    continue
                if not filename:
                    rejected_blocks.append(("<unknown>", "Empty filename in file marker"))
                    i = j + 1
                    continue
                if ensure_directory_exists(filename):
                    if filename in created_files or os.path.exists(filename):
                        overwrites.append(filename)
                        print(f"  Warning: Overwriting file: {filename}")
                    try:
                        with open(filename, 'w', encoding='utf-8') as output_file:
                            # Optionally remove trailing blank lines
                            while content_lines and content_lines[-1] == '':
                                content_lines.pop()
                            output_file.write('\n'.join(content_lines) + '\n')
                        created_files.add(filename)
                        print(f"Extracted: {filename}")
                        successful_extractions += 1
                    except Exception as e:
                        error_msg = f"Error writing file: {e}"
                        print(f"  Failed: {filename} - {error_msg}")
                        rejected_blocks.append((filename, error_msg))
                else:
                    rejected_blocks.append((filename, "Failed to create directory"))
                # Advance i to after the closing fence.
                i = j + 1
            else:
                i += 1

        return blocks_found, successful_extractions, rejected_blocks, overwrites

    except Exception as e:
        print(f"Error processing input file: {e}")
        sys.exit(1)

def main():
    # Get input file path.
    input_file_path = get_input_arguments()
    print(f"Processing compacted file: {input_file_path}")

    # Validate input file.
    is_valid, result = is_readable_text_file(input_file_path)
    if not is_valid:
        print(f"Error: {result}")
        sys.exit(1)
    encoding = result
    print(f"Input file encoding detected as: {encoding}")

    # Extract files.
    blocks_found, successful_extractions, rejected_blocks, overwrites = extract_files(input_file_path, encoding)

    # Print summary statistics.
    print("\nExtraction complete!")
    print(f"Total code blocks found: {blocks_found}")
    print(f"Files successfully extracted: {successful_extractions}")

    if rejected_blocks:
        print(f"Blocks rejected: {len(rejected_blocks)}")
        print("\nRejected blocks:")
        for block_id, reason in rejected_blocks:
            print(f"  - {block_id}: {reason}")

    if overwrites:
        print(f"\nFiles overwritten: {len(overwrites)}")
        print("Note: Multiple code blocks with the same filename were found; each file contains the content from the last matching block.")

if __name__ == "__main__":
    main()

