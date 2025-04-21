#!/usr/bin/env python

import os
import pandas as pd
import re

# Directory containing your daily CSVs
data_dir = "/gpfs1/home/p/w/pwormser/scratch/research/Outputs/730Days"
output = "/gpfs1/home/p/w/pwormser/scratch/research/Outputs"
# Pattern to match only the tweets_per_day_per_block files
pattern = re.compile(r"tweets_per_day_per_block_\d{4}-\d{2}-\d{2}\.csv")

# Accumulate all per-day data
all_blocks = []

for filename in os.listdir(data_dir):
    if pattern.match(filename):
        file_path = os.path.join(data_dir, filename)
        try:
            df = pd.read_csv(file_path, dtype={'GEOID10': str})
            all_blocks.append(df[['GEOID10', 'tweet_count']])
        except Exception as e:
            print(f"Error reading {filename}: {e}")

# Combine and group by GEOID10
if all_blocks:
    combined = pd.concat(all_blocks, ignore_index=True)
    summary = combined.groupby('GEOID10', as_index=False)['tweet_count'].sum()
    summary = summary.rename(columns={'tweet_count': 'Total_Tweets'})

    # Save result
    output_path = os.path.join(data_dir, "total_tweets_per_block.csv")
    summary.to_csv(output_path, index=False)
    print(f"Saved total tweets per block to {output_path}")
else:
    print("No matching files found.")

