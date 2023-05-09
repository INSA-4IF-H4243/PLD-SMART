import pandas as pd
import os

list_df = []
list_path = [path for path in os.listdir('.') if os.path.isfile(path) and path.endswith('.csv')]
big_df = pd.DataFrame()

for path in list_path:
    df = pd.read_csv(path)
    list_df.append(df)

big_df = pd.concat(list_df, ignore_index=True)
# big_df.to_csv('big_df.csv', index=False)

    