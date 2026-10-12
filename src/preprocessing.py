
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import re

# ==========================================
# 1. LOAD AND VALIDATE DATASET
# ==========================================

DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "llm_responses_dataset.csv"
)

df = pd.read_csv(DATA_FILE)

df.rename(
    columns={
        "llm_name": "LLM_name",
        "llm_input": "LLM_Input",
        "llm_output": "LLM_output",
    },
    inplace=True
)

required_columns = [
    "LLM_name", "prompt_id", "category",
    "LLM_Input", "LLM_output"
]

if df[required_columns].isna().any().any():
    raise ValueError("Dataset contains missing values.")

for column in ["LLM_name", "category", "LLM_Input", "LLM_output"]:
    df[column] = df[column].astype(str).str.strip()

df["prompt_id"] = df["prompt_id"].astype(int)

LABELS = {"GPT": 0, "Claude": 1, "Gemini": 2}
df["label"] = df["LLM_name"].map(LABELS)

if df["label"].isna().any():
    raise ValueError("Unknown LLM names found.")

if df.duplicated(["prompt_id", "LLM_name"]).any():
    raise ValueError("Duplicate prompt/model combinations found.")

# Every prompt should have exactly one response from each model.
for prompt_id, group in df.groupby("prompt_id"):
    if set(group["LLM_name"]) != set(LABELS):
        raise ValueError(
            f"Prompt {prompt_id} is missing one or more models."
        )

    if group["LLM_Input"].nunique() != 1:
        raise ValueError(
            f"Prompt {prompt_id} has inconsistent input text."
        )

print("=== DATASET ===")
print("Shape:", df.shape)
print(df["LLM_name"].value_counts())


# ==========================================
# 2. TOKENIZATION AND VOCABULARY
# ==========================================

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"

MAX_LENGTH = 200


def tokenize(text):
    """Convert text to lowercase word and punctuation tokens."""
    return re.findall(r"\b\w+\b|[^\w\s]", text.lower())


def build_vocabulary(texts):
    """Build vocabulary using training data only."""

    counts = Counter(
        token
        for text in texts
        for token in tokenize(text)
    )

    vocabulary = {
        PAD_TOKEN: 0,
        UNK_TOKEN: 1
    }

    for token, count in counts.most_common():
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary


def encode_and_pad(texts, vocabulary, max_length=MAX_LENGTH):
    """Convert text into fixed-length integer sequences."""

    encoded = []

    for text in texts:
        token_ids = [
            vocabulary.get(token, vocabulary[UNK_TOKEN])
            for token in tokenize(text)
        ]

        token_ids = token_ids[:max_length]

        token_ids += [vocabulary[PAD_TOKEN]] * (
            max_length - len(token_ids)
        )

        encoded.append(token_ids)

    return encoded


# ==========================================
# 3. SPLIT BY PROMPT ID
# ==========================================

prompt_ids = sorted(df["prompt_id"].unique())

train_ids, test_ids = train_test_split(
    prompt_ids,
    test_size=0.20,
    random_state=42
)

train_ids, val_ids = train_test_split(
    train_ids,
    test_size=0.20,
    random_state=42
)

train_df = df[df["prompt_id"].isin(train_ids)].copy()
val_df = df[df["prompt_id"].isin(val_ids)].copy()
test_df = df[df["prompt_id"].isin(test_ids)].copy()

assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)

print("\n=== DATA SPLIT ===")
print("Training:", len(train_df))
print("Validation:", len(val_df))
print("Testing:", len(test_df))


# ==========================================
# 4. SELECT TEXT CONFIGURATION
# ==========================================

def get_texts(dataframe, mode):
    """
    mode:
        input    = original prompt only
        output   = LLM response only
        combined = prompt + LLM response
    """

    if mode == "input":
        return dataframe["LLM_Input"].tolist()

    elif mode == "output":
        return dataframe["LLM_output"].tolist()

    elif mode == "combined":
        return (
            dataframe["LLM_Input"]
            + " "
            + dataframe["LLM_output"]
        ).tolist()

    else:
        raise ValueError(
            "Mode must be: input, output, or combined"
        )


# ==========================================
# 5. PREPARE EXPERIMENT
# ==========================================

def prepare_experiment(mode="output"):
    """
    Prepare one input configuration.

    Returns:
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        vocabulary
    """

    train_texts = get_texts(train_df, mode)
    val_texts = get_texts(val_df, mode)
    test_texts = get_texts(test_df, mode)

    # Vocabulary is built ONLY from training text.
    vocabulary = build_vocabulary(train_texts)

    X_train = encode_and_pad(train_texts, vocabulary)
    X_val = encode_and_pad(val_texts, vocabulary)
    X_test = encode_and_pad(test_texts, vocabulary)

    y_train = train_df["label"].astype(int).tolist()
    y_val = val_df["label"].astype(int).tolist()
    y_test = test_df["label"].astype(int).tolist()

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        vocabulary
    )


# ==========================================
# 6. DEFAULT OUTPUT-ONLY EXPERIMENT (RQ1)
# ==========================================

(
    X_train_output,
    X_val_output,
    X_test_output,
    y_train,
    y_val,
    y_test,
    vocab
) = prepare_experiment("output")

print("\n=== DEFAULT EXPERIMENT: OUTPUT ONLY ===")
print("Vocabulary size:", len(vocab))
print("Training sequences:", len(X_train_output))
print("Validation sequences:", len(X_val_output))
print("Testing sequences:", len(X_test_output))


# ==========================================
# 7. OPTIONAL: DATASET EXPLORATION
# ==========================================

if __name__ == "__main__":

    print("\n=== AVAILABLE EXPERIMENTS ===")

    for mode in ["input", "output", "combined"]:

        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            vocabulary
        ) = prepare_experiment(mode)

        print(f"\nMode: {mode}")
        print("Vocabulary size:", len(vocabulary))
        print("Train:", len(X_train))
        print("Validation:", len(X_val))
        print("Test:", len(X_test))
        print("Sequence length:", MAX_LENGTH)
