import pandas as pd

# Load CSV
df = pd.read_csv("train.csv")

# Filter English
df_en = df[df['language'] == 'English'].copy()

# Save
df_en.to_csv("english_full_text.csv", index=False)

print(df_en.head())