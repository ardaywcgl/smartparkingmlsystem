import pandas as pd

df = pd.read_csv("dataset.csv")

df["LastUpdated"] = pd.to_datetime(df["LastUpdated"])
df["hour"] = df["LastUpdated"].dt.hour

print(df.groupby("hour")["Occupancy"].mean())