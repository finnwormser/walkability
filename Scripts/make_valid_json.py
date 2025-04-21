import csv
import ast
import json
from tqdm.auto import tqdm

INFILE  = '/users/p/w/pwormser/scratch/research/Outputs/walkability_data_with_counters.csv'
OUTFILE = '/users/p/w/pwormser/scratch/research/Outputs/walkability_data_with_counters.clean.csv'

with open(INFILE, newline='') as fin, open(OUTFILE, 'w', newline='') as fout:
    reader = csv.DictReader(fin)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(fout, fieldnames=fieldnames)
    writer.writeheader()

    for row in tqdm(reader, desc="Cleaning counters → JSON"):
        try:
            # Safely convert from single-quoted string to Python dict
            parsed_dict = ast.literal_eval(row['counters'])

            # Convert to compact valid JSON string
            row['counters'] = json.dumps(parsed_dict, separators=(",", ":"))

            writer.writerow(row)

        except Exception as e:
            print(f"Skipping row due to error: {e}")
            continue

print("✅ Cleaned CSV saved to:", OUTFILE)

