import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text
import seaborn as sns

datadir = Path('/gpfs1/home/p/w/pwormser/scratch/research/Outputs/730Days')
smartlocdir = Path('/users/p/w/pwormser/scratch/research/DataSources')
df_smartloc = pd.read_csv(smartlocdir / 'SmartLocationData.csv')

outputdir = Path.cwd()
outputdb = outputdir / 'walkability_data_with_counters.db'
print(f'sqlite:///{str(outputdb)}')

files_agg = datadir.glob('agg*.csv')
files_agg = sorted([f for f in files_agg])

def get_date_from_filename(filepath):
    stem = filepath.stem
    date = stem.split('_')[-1]

    return date

def write_to_db(overwrite=False):
    file_subset = files_agg
    # if outputdb.exists():
    #     if overwrite:
    #         outputdb.unlink() # delete the test.db file to ensure we write fresh
    #     else:
    #         raise Exception(f'File {outputdb} exists. Set overwrite=True.')
    engine = create_engine(f'sqlite:///{str(outputdb)}')
    df_smartloc.to_sql(name='smartloc', con=engine, chunksize=1000)

    for f in file_subset:
        print('Reading...')
        df = pd.read_csv(f, 
                         # usecols=['GEOID10', 'tweet_count', 'sentiment', 'sentiment_std']
                        )
        date = get_date_from_filename(f)
        df['date'] = date
        assert df[['GEOID10', 'date']].duplicated().sum() == 0
        # df = df.merge(df_smartloc[['GEOID10', 'CSA_Name', 'CBSA_Name']], on='GEOID10', how='left')
        df = df.set_index(['GEOID10', 'date'])

        print('Writing...')
        df.to_sql(name='tweet_sent_per_dayblock', con=engine, if_exists='append', index=True)

        print(f'\t{f.name}')
        # print(df.head())
    
    # return file_subset

write_to_db(overwrite=False)