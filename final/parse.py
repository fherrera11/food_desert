import pandas as pd

df = pd.read_csv("food_inspection_table.csv")

df = df[df["City"] == "Merced"]

df = df.drop(columns="Detail")

df.to_csv("output.csv", index=False)
