import re

# Read input strings from file
with open('additional/vgg16adam.txt', 'r', encoding='utf-8') as f:
    input_strings = f.readlines()

# Extract numeric values from strings
numeric_values = []
for input_string in input_strings:
    match = re.search(r'tensor\((.*?)\)', input_string)
    if match:
        numeric_values.append(match.group(1))

# Write numeric values to file
with open('additional/f1vgg16adam.txt', 'w', encoding='utf-8') as f:
    for numeric_value in numeric_values:
        f.write(numeric_value + '\n')
