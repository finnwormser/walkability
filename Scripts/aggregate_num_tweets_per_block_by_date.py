import os
import glob

input_folder = '/gpfs1/home/p/w/pwormser/scratch/research/Outputs730Days'
output_file = '/gpfs1/home/p/w/pwormser/scratch/research/Outputs/aggregated_tweets_all_dates.csv'

# Only include files that have "aggregated" in their name.
files = [f for f in glob.glob(os.path.join(input_folder, '*.csv')) if "aggregated" in os.path.basename(f)]

with open(output_file, 'w') as outfile:
    # Write the header for the new CSV:
    outfile.write("GEOID10,tweet_count,sentiment,sentiment_std,date\n")
    
    for file_path in files:
        basename = os.path.basename(file_path)
        # A simple approach: split filename on '_' and '.' to extract the date.
        # Expecting something like: aggregated_tweets_2015-04-27.csv
        parts = basename.split('_')
        if len(parts) < 3:
            print("Skipping file (unexpected format):", basename)
            continue
        # parts[2] is expected to have date + .csv, so remove the '.csv'
        date_str = parts[2].replace('.csv', '')
        
        # Print for debugging
        print(f"Processing file {basename} with date {date_str}")
        
        with open(file_path, 'r') as infile:
            # Read and discard the header line
            header_line = infile.readline()
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                fields = line.split(',')
                if len(fields) < 5:
                    continue
                GEOID10 = fields[0]
                tweet_count = fields[2]
                sentiment = fields[3]
                sentiment_std = fields[4]
                outfile.write(f"{GEOID10},{tweet_count},{sentiment},{sentiment_std},{date_str}\n")

print("Done! Output saved to:", output_file)
