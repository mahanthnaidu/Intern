import csv

# Define the input and output CSV file paths
input_csv_file_path = 'input_data3.csv'
output_csv_file_path = 'output_data3.csv'

# Read data from the input CSV file
with open(input_csv_file_path, mode='r', newline='') as infile:
    reader = csv.reader(infile)
    data = list(reader)

# Replace empty values with zeros
data = [[cell if cell else '0' for cell in row] for row in data]

# Write the cleaned data to the output CSV file

with open(output_csv_file_path, mode='w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerows(data)

print(f"Data successfully written to {output_csv_file_path}")
