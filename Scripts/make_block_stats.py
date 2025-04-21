#!/usr/bin/env python3
"""
This script reads from a source SQLite database file and computes statistics for each unique GEOID10.
The statistics include:
  1. Maximum tweet_count value
  2. Minimum tweet_count value
  3. Row count for that GEOID10
  4. Mean tweet_count
  5. Median tweet_count
  6. Standard deviation of tweet_count

Results are written to a CSV file.

Usage:
    python generate_stats.py

Modify the variables SOURCE_DB, TARGET_CSV, and TABLE_NAME if needed.
"""

import sqlite3
import csv
import statistics
from collections import defaultdict

# Define your file and table names.
SOURCE_DB = '/users/p/w/pwormser/scratch/research/Outputs/walkability_data_with_counters_slurm.db'         
TABLE_NAME = 'tweet_sent_per_dayblock' 
TARGET_CSV = 'block_stats.csv'       

def fetch_data(db_file, table):
    """Connect to the database and fetch GEOID10 and tweet_count columns."""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    try:
        query = f"SELECT GEOID10, tweet_count FROM {table};"
        cursor.execute(query)
        rows = cursor.fetchall()
    except sqlite3.Error as e:
        print("SQLite error:", e)
        rows = []
    finally:
        conn.close()
    return rows

def compute_statistics(rows):
    """Compute statistics for each unique GEOID10 from the list of rows."""
    # Group tweet_count values by GEOID10.
    data_by_geo = defaultdict(list)
    for geoid, tweet_count in rows:
        data_by_geo[geoid].append(tweet_count)

    results = []
    for geoid, counts in data_by_geo.items():
        max_tweets = max(counts)
        min_tweets = min(counts)
        row_count = len(counts)
        mean_tweets = sum(counts) / row_count
        median_tweets = statistics.median(counts)
        # Standard deviation requires at least two data points.
        stdev_tweets = statistics.stdev(counts) if row_count > 1 else 0.0

        results.append({
            'GEOID10': geoid,
            'max_tweets': max_tweets,
            'min_tweets': min_tweets,
            'row_count': row_count,
            'mean_tweets': mean_tweets,
            'median_tweets': median_tweets,
            'stdev_tweets': stdev_tweets
        })
    return results

def write_csv(results, csv_file):
    """Write the computed statistics to a CSV file."""
    headers = ["GEOID10", "max_tweets", "min_tweets", "row_count", "mean_tweets", "median_tweets", "stdev_tweets"]
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(results)
    print(f"Statistics have been written to {csv_file}")

def main():
    rows = fetch_data(SOURCE_DB, TABLE_NAME)
    if not rows:
        print("No data found or there was an error fetching data from the table.")
        return

    stats = compute_statistics(rows)
    write_csv(stats, TARGET_CSV)

if __name__ == "__main__":
    main()

