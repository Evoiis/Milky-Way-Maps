from query import GaiaQueryWrapper
import pandas as pd
import os

file_name = "100k_analysis_data.csv"

if os.path.exists(file_name):
    df = pd.read_csv(file_name)
else:
    gqw = GaiaQueryWrapper()
    df : pd.DataFrame = gqw.get_gaia_data()
    df.to_csv(file_name)

print(f"{df.isna().sum()=}")

print(f"{len(df)=}")

