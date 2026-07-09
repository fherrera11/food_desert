import pandas as pd

df = pd.read_csv("geocode_address.csv")


rating_map = {
    "Good":2,
    "Satisfactory":1,
    "Unsatisfactory":0
}

df['quality_score'] = df["USER_Current_Rating"].map(rating_map).fillna(0).astype(int)

df.to_csv("cleaned_joined_output.csv", index=False)