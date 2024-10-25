import json

# Specify your input and output file names
input_file = 'data/yelp_dataset/yelp_academic_dataset_review.json'
output_file = 'data/yelp_dataset/truncated_yelp_academic_dataset_review.json'

# Initialize an empty list to hold the truncated data
truncated_data = []

with open(input_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= 1000000:
            break
        truncated_data.append(json.loads(line))  # Load each line as a JSON object

# Write the truncated data to a new file
with open(output_file, 'w') as f:
    json.dump(truncated_data, f)  # Dump with pretty formatting