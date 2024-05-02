import os

def count_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return sum(1 for line in file if line.strip() and not line.strip().startswith('#'))

def count_lines_in_directory(directory):
    total_lines = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_lines += count_lines(file_path)
    return total_lines

if __name__ == "__main__":
    directory = input("Enter directory path: ")
    lines_of_code = count_lines_in_directory(directory)
    print(f"Total lines of executable code: {lines_of_code}")
