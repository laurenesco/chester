import os

# Function to read file contents and write to output file
def read_files(directory, output_file):
    with open(output_file, 'a', encoding='utf-8') as out_file:
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath):
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                    out_file.write(f"File: {filepath}\n")
                    for i, line in enumerate(file, start=1):
                        out_file.write(f"{i}: {line}")
                    out_file.write("\n\n")

# Main function
def main():
    # Current directory
    current_dir = os.getcwd()
    output_file = "filecontents.txt"

    # Process files in the current directory
    read_files(current_dir, output_file)

    # Directories to process
    specified_directories = [
        "C:/Users/laesc/OneDrive/Desktop/chester/chess_classes",
        "C:/Users/laesc/OneDrive/Desktop/chester/chess_classes/piece_classes",
        "C:/Users/laesc/OneDrive/Desktop/chester/env",
        "C:/Users/laesc/OneDrive/Desktop/chester/ml",
        "C:/Users/laesc/OneDrive/Desktop/chester/python",
        "C:/Users/laesc/OneDrive/Desktop/chester/screens",
        "C:/Users/laesc/OneDrive/Desktop/chester/scripts",
        "C:/Users/laesc/OneDrive/Desktop/chester/styling"
        # Add more directories here if needed
    ]

    # Process specified directories
    for directory in specified_directories:
        if os.path.isdir(directory):
            read_files(directory, output_file)
        else:
            print(f"Directory '{directory}' not found.")

if __name__ == "__main__":
    main()
