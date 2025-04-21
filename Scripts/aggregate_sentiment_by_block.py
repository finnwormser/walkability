import os
import glob
import pandas as pd

def main():
    # Define the file pattern to match the aggregated tweets CSV files.
    data_dir = "/gpfs1/home/p/w/pwormser/scratch/research/730Days"  # change this if your folder has a different name
    input_pattern = os.path.join(data_dir, "aggregated_tweets_*.csv")
    all_files = glob.glob(input_pattern)
    print(f"Found {len(all_files)} files matching the pattern '{input_pattern}'.")

    if not all_files:
        print("No files found. Please check the file pattern or the directory.")
        return

    # List to collect dataframes from each file.
    dfs = []
    
    for file in all_files:
        try:
            df = pd.read_csv(file)
            # Ensure the required columns exist.
            required_columns = {'GEOID10', 'tweet_count', 'sentiment', 'sentiment_std'}
            if required_columns.issubset(df.columns):
                # Append only the necessary columns.
                dfs.append(df[['GEOID10', 'tweet_count', 'sentiment', 'sentiment_std']])
            else:
                print(f"Skipping {file}: required columns missing.")
        except Exception as e:
            print(f"Error reading {file}: {e}")

    if not dfs:
        print("No valid files to process.")
        return

    def weighted_sentiment(group):
        tw = group['tweet_count']
        s  = group['sentiment']
        return (s * tw).sum() / tw.sum()

    # Concatenate all the data into one DataFrame.
    all_data = pd.concat(dfs, ignore_index=True)

    # Group by GEOID10 and aggregate:
    # - Sum tweet_count.
    # - Mean sentiment and sentiment_std.
   aggregated = (
    all_data
      .groupby('GEOID10')
      .agg(
        tweet_count      = ('tweet_count', 'sum'),
        sentiment_weighted = (weighted_sentiment, lambda x: x),  # or use .apply
        sentiment_std    = ('sentiment_std', 'mean')
      )
      .reset_index()
	) 

    # Save the aggregated data to a CSV file.
    output_file = "aggregated_all.csv"
    aggregated.to_csv(output_file, index=False)
    print(f"Aggregated data saved to {output_file}")

if __name__ == "__main__":
    main()

