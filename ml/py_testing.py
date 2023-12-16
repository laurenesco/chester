# import tensorflow as tf

try:
    with open("xor.txt", "r") as file:
        data = file.read()
        substrings = data.splitlines()

        for substring in substrings:
            print(substring)
except Exception as e:
    print(e)
