import pandas as pd
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "llm_responses_dataset.csv"

df = pd.read_csv(DATA_FILE)

# Accept the lowercase headers used by the current dataset.
df.rename(
    columns={
        "llm_name": "LLM_name",
        "llm_input": "LLM_Input",
        "llm_output": "LLM_output",
    },
    inplace=True
)

# Remove accidental whitespace surrounding basic metadata
df["LLM_name"] = df["LLM_name"].str.strip()
df["category"] = df["category"].str.strip()
df["LLM_Input"] = df["LLM_Input"].str.strip()
df["LLM_output"] = df["LLM_output"].str.strip()

# Convert prompt IDs to integers
df["prompt_id"] = df["prompt_id"].astype(int)

print(df.head())
print("\nDataset shape:", df.shape)
print("\nResponses per model:")
print(df["LLM_name"].value_counts())