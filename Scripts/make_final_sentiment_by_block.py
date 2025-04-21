import csv, math, orjson, multiprocessing as mp
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import ast

from data_mountain_query.counters  import NgramCounter, lang_dict
from data_mountain_query.sentiment import load_happs_scores, counter_sentiment
import csv
import sys

# Increase CSV field size limit
csv.field_size_limit(sys.maxsize)

# -------- parameters --------
INFILE = '/gpfs2/scratch/pwormser/research/Outputs/walkability_data_with_counters.csv'
N_GROUPS = 4
LEXICON  = load_happs_scores(lang=lang_dict["en"])

# -------- helpers --------
def date_to_group(d: datetime) -> int:
    """Map a date to 0…N_GROUPS‑1 (730 days assumed)."""
    day_idx = (d - START_DATE).days           # 0 … 729
    return min(N_GROUPS-1, day_idx * N_GROUPS // TOTAL_DAYS)

def merge_into(target, src):
    """In‑place += of nested dicts."""
    for w, c in src.items():
        slot = target.setdefault(w, {"count":0, "count_no_rt":0})
        slot["count"]      += c.get("count",0)
        slot["count_no_rt"]+= c.get("count_no_rt",0)

def score_chunk(item):
    """Worker: (geo, merged_dicts, tweet_sum_per_group) → (geo, μ, σ, twts)"""
    geo, merged_dicts, twt_per_grp = item
    sents, stds = [], []
    for d in merged_dicts:
        nc = NgramCounter(d)
        sc, _, sd = counter_sentiment(nc, LEXICON, None)
        if sc is not None:
            sents.append(sc);  stds.append(sd)
    if not sents:
        return None                      # drop this GEOID10
    return (
        geo,
        float(np.mean(sents)),
        float(np.mean(stds)),
        int(sum(twt_per_grp))
    )

# First pass to find min/max dates
with open(INFILE, newline='') as fh:
    rdr = csv.DictReader(fh)
    dates = [datetime.fromisoformat(r["date"]) for r in rdr]
START_DATE = min(dates)
END_DATE   = max(dates)
TOTAL_DAYS = (END_DATE - START_DATE).days + 1

# -------- streaming aggregation --------
geo_grp_counters = defaultdict(lambda: [defaultdict(dict) for _ in range(N_GROUPS)])
geo_grp_tweets   = defaultdict(lambda: [0]*N_GROUPS)

with open(INFILE, newline='') as fh:
    reader = csv.DictReader(fh)

    for row in reader:
        geo = row["GEOID10"]
        grp = date_to_group(datetime.fromisoformat(row["date"]))
    
        # instead of orjson:
        cnt = ast.literal_eval(row["counters"])
        merge_into(geo_grp_counters[geo][grp], cnt)
        geo_grp_tweets  [geo][grp] += int(row["tweet_count"])

# -------- parallel sentiment scoring --------
if __name__ == "__main__":
    pool = mp.Pool()
    payload = [
        (geo, merged_list, geo_grp_tweets[geo])
        for geo, merged_list in geo_grp_counters.items()
    ]
    results = []
    for res in tqdm(pool.imap_unordered(score_chunk, payload, chunksize=500),
                    total=len(payload), desc="Scoring"):
        if res:
            results.append(res)
    pool.close(); pool.join()
    
    # -------- output --------
    out_df = pd.DataFrame(
        results, columns=["GEOID10","sentiment","sentiment_std","tweet_count"]
    )
    out_df.to_csv(f"all_bgs_aggregated_sentiment_{N_GROUPS}_periods.csv", index=False)
    print("Saved", len(out_df), "rows.")

