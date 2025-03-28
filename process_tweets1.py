#!/usr/bin/env python

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import os
import fiona
from data_mountain_query.connection import get_connection
from data_mountain_query.query import get_tweets
from data_mountain_query.sentiment import load_happs_scores, df_sentiment
from data_mountain_query.parsers import load_ngrams_parser, parse_ngrams_tweet
from data_mountain_query.counters import lang_dict

# ----------------------------------------------------------------------
# Global setting: whether to record presence only (each word counts as 1 per tweet)
presence_only = False

# ----------------------------------------------------------------------
# Load Block Groups with caching.
gdb_path = "/gpfs2/scratch/pwormser/research/DataSources/SmartLocationDatabase_copy.gdb"
cache_file = "block_groups.pkl"
if os.path.exists(cache_file):
    block_groups = gpd.GeoDataFrame(pd.read_pickle(cache_file))
    print("Loaded block_groups from cache.")
else:
    layers = fiona.listlayers(gdb_path)
    print("Available layers:", layers)
    block_groups = gpd.read_file(gdb_path, layer="EPA_SLD_Database_V3")
    block_groups = block_groups.to_crs("EPSG:4326")
    block_groups.to_pickle(cache_file)
    print("Read block_groups from GDB and cached it.")

# ----------------------------------------------------------------------
# Global: Load US Boundary.
us_boundary = gpd.read_file('/gpfs2/scratch/pwormser/research/DataSources/cb_2018_us_nation_20m/cb_2018_us_nation_20m.shp')
us_boundary = us_boundary.to_crs("EPSG:4326")

# ----------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------
def extract_lat_lon(df):
    """
    Extracts latitude and longitude from the 'geo' column using a vectorized method.
    Resets the index so that the join aligns properly.
    """
    df = df.reset_index(drop=True)
    coords = pd.DataFrame(
        df['geo'].apply(lambda g: g.get('coordinates', [None, None]) if isinstance(g, dict) else [None, None]).tolist(),
        columns=['longitude', 'latitude']
    )
    # Convert to numeric values.
    coords['longitude'] = pd.to_numeric(coords['longitude'], errors='coerce')
    coords['latitude'] = pd.to_numeric(coords['latitude'], errors='coerce')
    coords = coords.reset_index(drop=True)
    return df.join(coords)

def filter_us_tweets(df):
    """
    Filters tweets to only those within defined US bounding boxes.
    """
    continental = ((df['latitude'] >= 24.396308) & (df['latitude'] <= 49.384358) &
                   (df['longitude'] >= -125.0) & (df['longitude'] <= -66.93457))
    alaska = ((df['latitude'] >= 51.2) & (df['latitude'] <= 71.4) &
              (df['longitude'] >= -179.1) & (df['longitude'] <= -129.9))
    hawaii = ((df['latitude'] >= 18.9) & (df['latitude'] <= 22.2) &
              (df['longitude'] >= -160.3) & (df['longitude'] <= -154.8))
    us_mask = continental | alaska | hawaii
    return df[us_mask].copy()

def refine_with_us_boundary(df):
    """
    Filters tweets to only those within the US boundary using a spatial join.
    """
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")
    gdf = gdf.to_crs(block_groups.crs)
    us_boundary_reproj = us_boundary.to_crs(gdf.crs)
    joined = gpd.sjoin(gdf, us_boundary_reproj, how="inner", predicate="within")
    joined = joined.reset_index(drop=True)
    return joined.drop(columns=["geometry", "index_right"], errors='ignore')

def tag_tweets_with_geodata(df):
    """
    Attaches census block (GEOID10) and ZIP code (ZIP_Code) information to tweets.
    (Here we only attach GEOID10, since our primary grouping is per block.)
    """
    target_crs = block_groups.crs
    zip_shp = "/gpfs2/scratch/pwormser/research/DataSources/USA_Boundaries_2022_-232574676275878974/USA_ZipCode.shp"
    zip_codes_local = gpd.read_file(zip_shp).to_crs(target_crs)
    if 'index_right' in zip_codes_local.columns:
        zip_codes_local.drop(columns=['index_right'], inplace=True)
    
    bg = block_groups.copy()
    if 'index_right' in bg.columns:
        bg.drop(columns=['index_right'], inplace=True)
    
    id_column = 'GEOID10' if 'GEOID10' in bg.columns else 'GEOID'
    
    points_gdf = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df['longitude'], df['latitude'])]
    , crs="EPSG:4326").to_crs(target_crs)
    if 'index_right' in points_gdf.columns:
        points_gdf.drop(columns=['index_right'], inplace=True)
    
    points_bg = gpd.sjoin(points_gdf, bg[[id_column, 'geometry']], how="left", predicate="within")
    if 'index_right' in points_bg.columns:
        points_bg.drop(columns=['index_right'], inplace=True)
    if id_column != 'GEOID10':
        points_bg = points_bg.rename(columns={id_column: 'GEOID10'})
    
    df['GEOID10'] = points_bg['GEOID10'].astype("category")
    return df

def merge_counters(counter_list):
    """
    Merges a list of tweet counter dictionaries.
    """
    aggregated = {}
    for counter in counter_list:
        for word, counts in counter.items():
            if word not in aggregated:
                aggregated[word] = counts.copy()
            else:
                for key, value in counts.items():
                    aggregated[word][key] = aggregated[word].get(key, 0) + value
    return aggregated

def aggregate_counters_by_block(df):
    """
    Aggregates the 'counters' dictionaries for each census block group.
    Expects a DataFrame with 'GEOID10' and 'counters' columns.
    """
    df = df.copy()
    df.loc[:, 'counters'] = df['counters'].apply(lambda x: x if isinstance(x, dict) else {})
    grouped = df.groupby('GEOID10', as_index=False, observed=False)['counters'].agg(
        lambda counters: merge_counters(list(counters))
    )
    grouped.columns = ['GEOID10', 'aggregated_counters']
    return grouped
    
def process_chunk(chunk_df):
    # Force re-extraction of latitude/longitude.
    chunk_df = chunk_df.drop(columns=['latitude', 'longitude'], errors='ignore')
    chunk_df = extract_lat_lon(chunk_df)
    chunk_df = filter_us_tweets(chunk_df)
    chunk_df = refine_with_us_boundary(chunk_df)
    # Processed DataFrame with GEOID10 added.
    processed_df = tag_tweets_with_geodata(chunk_df)
    
    # Compute aggregated counters for the chunk.
    agg_df = aggregate_counters_by_block(processed_df)
    agg_df = agg_df.reset_index(drop=True)
    
    # Do not merge tweet counts; just keep aggregated counters.
    tweet_counts = processed_df.groupby('GEOID10', observed=False).size().reset_index(name='tweet_count')
    agg_df = pd.merge(agg_df, tweet_counts, on='GEOID10', how='left')
    aggregated_result = {row['GEOID10']: {"aggregated_counters": row['aggregated_counters'],
                                           "tweet_count": row['tweet_count']} for _, row in agg_df.iterrows()}
    
    # Compute daily counts on the processed DataFrame.
    processed_df['Date'] = pd.to_datetime(processed_df["tweet_created_at"]).dt.date
    daily_chunk = processed_df.groupby(['Date', 'GEOID10'], observed=False).size().reset_index(name='tweet_count')
    
    return aggregated_result, daily_chunk



# ------------------------------------------------------------------------------
# Main Processing Pipeline
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Read 30 dates from a CSV file.
    input_csv = "/gpfs2/scratch/pwormser/research/DataSources/15_shuffled_dates_2014.csv"
    
    date_df = pd.read_csv(input_csv, parse_dates=["Date"])
    dates = date_df["Date"].tolist()
    
    lang = "en"
    collection, client = get_connection(geotweets=True)
    
    # Prepare a list to hold DataFrames for each day.
    tweet_dfs = []

    for current_date in dates:
        start_date = current_date
        end_date = current_date + timedelta(days=1)
        print(f"Fetching tweets for {start_date.date()}...")
        query = {
            "tweet_created_at": {"$gte": start_date, "$lt": end_date},
            "fastText_lang": lang,
        }
        
        tweets = get_tweets(collection, query, limit=0)
        ngrams_parser = load_ngrams_parser()
        tweets = [{**tweet, "counters": parse_ngrams_tweet(tweet, ngrams_parser)} for tweet in tweets]
        if presence_only:
            tweets = [{**tweet, "counters": {w: {'count': 1, 'count_no_rt': 1} for w in tweet.get("counters", {})}} for tweet in tweets]
        df_day = pd.DataFrame(tweets)
        
        if 'tweet_created_at' in df_day.columns:
            df_day["Date"] = pd.to_datetime(df_day["tweet_created_at"]).dt.normalize()
        tweet_dfs.append(df_day)
    
    if tweet_dfs:
        df_all = pd.concat(tweet_dfs, ignore_index=True)
    else:
        df_all = pd.DataFrame()

        
    
        # Process tweets in chunks.
    chunk_size = 300  # Adjust as needed.
    all_agg = {}
    daily_counts_list = []  # For per-day, per-block tweet counts.
    
    for i in range(0, len(df_all), chunk_size):
        chunk_df = df_all.iloc[i : i + chunk_size].copy()
        agg_dict, daily_chunk = process_chunk(chunk_df)
        daily_counts_list.append(daily_chunk)
        
        for geoid, result in agg_dict.items():
            if geoid in all_agg:
                all_agg[geoid]["aggregated_counters"] = merge_counters([all_agg[geoid]["aggregated_counters"],
                                                                        result["aggregated_counters"]])
                all_agg[geoid]["tweet_count"] += result["tweet_count"]
            else:
                all_agg[geoid] = result
    
    # Combine daily counts across chunks and sum tweet counts for each (Date, GEOID10) pair.
    daily_counts_df = pd.concat(daily_counts_list, ignore_index=True)
    daily_counts_df = daily_counts_df.groupby(['Date', 'GEOID10'], as_index=False)['tweet_count'].sum()
    
    final_agg = pd.DataFrame({
        'GEOID10': list(all_agg.keys()),
        'aggregated_counters': [res["aggregated_counters"] for res in all_agg.values()],
        'tweet_count': [res["tweet_count"] for res in all_agg.values()]
    })

    final_agg = final_agg.rename(columns={'aggregated_counters': 'counters'})
    word2score = load_happs_scores(lang=lang_dict[lang])
    final_agg = df_sentiment(final_agg, word2score=word2score)

    # Save output CSV files.
    final_agg.to_csv("aggregated_tweets_30days_long.csv", index=False)
    final_agg[['GEOID10', 'tweet_count']].to_csv("tweet_counts_30days.csv", index=False)
    daily_counts_df.to_csv("tweets_per_day_per_block_30days.csv", index=False)
    
    print("Aggregated tweet data saved to aggregated_tweets_30days_long.csv")
    print("Overall tweet counts saved to tweet_counts_30days.csv")
    print("Daily tweet counts per block saved to tweets_per_day_per_block_30days.csv")
    print(final_agg.head())
    print(daily_counts_df.head())


