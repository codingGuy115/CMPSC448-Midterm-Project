import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter
import re

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


# --------------------------------------------------
# Step 2: Encode LLM labels
# --------------------------------------------------

label_map = {
    "GPT": 0,
    "Claude": 1,
    "Gemini": 2
}

# Convert each LLM name into its numerical label
df["label"] = df["LLM_name"].map(label_map)

# Check for unexpected LLM names
if df["label"].isnull().any():
    raise ValueError("Some LLM names could not be encoded.")

# Convert labels to integers
df["label"] = df["label"].astype(int)

# Verify the results
print("\nEncoded labels:")
print(df[["LLM_name", "label"]].drop_duplicates())

print("\nResponses per label:")
print(df["label"].value_counts().sort_index())

print("\nMissing labels:", df["label"].isnull().sum())


# --------------------------------------------------
# Step 3: Split into training, validation, and testing
# --------------------------------------------------

# Get all unique prompt IDs
prompt_ids = df["prompt_id"].unique()

# First split: 70% training, 30% remaining
train_ids, remaining_ids = train_test_split(
    prompt_ids,
    test_size=0.30,
    random_state=42,
    shuffle=True
)

# Second split: 7 prompts for validation,
# 8 prompts for testing
val_ids, test_ids = train_test_split(
    remaining_ids,
    test_size=8,
    random_state=42,
    shuffle=True
)

# Select the responses belonging to each split
train_df = df[df["prompt_id"].isin(train_ids)].copy()
val_df = df[df["prompt_id"].isin(val_ids)].copy()
test_df = df[df["prompt_id"].isin(test_ids)].copy()

# Reset row indices
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

# --------------------------------------------------
# Verify the results
# --------------------------------------------------

print("\n=== Dataset Splits ===")

print("Training responses:", len(train_df))
print("Validation responses:", len(val_df))
print("Testing responses:", len(test_df))

print("\nTraining labels:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation labels:")
print(val_df["label"].value_counts().sort_index())

print("\nTesting labels:")
print(test_df["label"].value_counts().sort_index())

# Check for overlapping prompt IDs
assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids).isdisjoint(test_ids)
assert set(val_ids).isdisjoint(test_ids)

# Check that every response belongs to exactly one split
assert len(train_df) + len(val_df) + len(test_df) == len(df)

# Check that each split contains all three models
for split_df in [train_df, val_df, test_df]:
    assert set(split_df["label"].unique()) == {0, 1, 2}

    counts = split_df["label"].value_counts()
    assert counts.nunique() == 1

print("\nPASS: No overlapping prompt IDs.")
print("PASS: All responses accounted for.")
print("PASS: All splits have balanced LLM labels.")

# --------------------------------------------------
# Step 4: Text tokenization
# --------------------------------------------------

def tokenize(text):
    text = str(text)

    # Keep [SEP] together as one token
    tokens = re.findall(r"\[SEP\]|\w+|[^\w\s]", text)

    return tokens



def prepare_text(df, mode="output"):
    if mode == "output":
        texts = df["LLM_output"].tolist()

    elif mode == "input":
        texts = df["LLM_Input"].tolist()

    elif mode == "both":
        texts = (
            df["LLM_Input"] + " [SEP] " + df["LLM_output"]
        ).tolist()

    else:
        raise ValueError("Mode must be 'input', 'output', or 'both'.")

    return [tokenize(text) for text in texts]


# Test all three configurations using training data
train_output_tokens = prepare_text(train_df, mode="output")
train_input_tokens = prepare_text(train_df, mode="input")
train_both_tokens = prepare_text(train_df, mode="both")

# --------------------------------------------------
# Step 5: Build vocabulary from training data only
# --------------------------------------------------

def build_vocabulary(tokenized_texts):
    # Count how frequently each token appears
    token_counts = Counter()

    for tokens in tokenized_texts:
        token_counts.update(tokens)

    # Reserve IDs for special tokens
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[SEP]": 2
    }

    # Add ordinary tokens to the vocabulary
    for token, count in sorted(token_counts.items()):
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)

    return vocabulary


# Build vocabulary using only training examples.
# Include both prompts and responses so it supports RQ1 and RQ2.
training_texts = train_input_tokens + train_output_tokens

vocab = build_vocabulary(training_texts)

# --------------------------------------------------
# Step 6: Numerical encoding and padding
# --------------------------------------------------

def encode_tokens(tokens, vocabulary):
    return [
        vocabulary.get(token, vocabulary["[UNK]"])
        for token in tokens
    ]


def pad_sequence(sequence, max_length, pad_id=0):
    # Cut off tokens if the sequence is too long
    sequence = sequence[:max_length]

    # Add padding if the sequence is too short
    padding_needed = max_length - len(sequence)

    sequence = sequence + [pad_id] * padding_needed

    return sequence


def encode_and_pad(tokenized_texts, vocabulary, max_length):
    processed_texts = []

    for tokens in tokenized_texts:
        encoded = encode_tokens(tokens, vocabulary)

        padded = pad_sequence(
            encoded,
            max_length,
            pad_id=vocabulary["[PAD]"]
        )

        processed_texts.append(padded)

    return processed_texts

# --------------------------------------------------
# Step 7: Examine sequence lengths
# --------------------------------------------------

output_lengths = [len(tokens) for tokens in train_output_tokens]
input_lengths = [len(tokens) for tokens in train_input_tokens]
both_lengths = [len(tokens) for tokens in train_both_tokens]

def show_length_stats(name, lengths):
    print(f"\n{name}:")
    print("Shortest:", min(lengths))
    print("Longest:", max(lengths))
    print("Average:", round(sum(lengths) / len(lengths), 2))


print("\n=== Training Sequence Lengths ===")

show_length_stats("Output-only", output_lengths)
show_length_stats("Input-only", input_lengths)
show_length_stats("Input + Output", both_lengths)


# --------------------------------------------------
# Step 8: Prepare numerical sequences for each split
# --------------------------------------------------

MAX_OUTPUT_LENGTH = 1024
MAX_INPUT_LENGTH = 64

def prepare_sequences(split_df, mode, max_length):
    tokenized = prepare_text(split_df, mode=mode)
    return encode_and_pad(tokenized, vocab, max_length)


# Output-only data for RQ1 and RQ2
X_train_output = prepare_sequences(
    train_df, "output", MAX_OUTPUT_LENGTH
)
X_val_output = prepare_sequences(
    val_df, "output", MAX_OUTPUT_LENGTH
)
X_test_output = prepare_sequences(
    test_df, "output", MAX_OUTPUT_LENGTH
)

# Input-only data for RQ2
X_train_input = prepare_sequences(
    train_df, "input", MAX_INPUT_LENGTH
)
X_val_input = prepare_sequences(
    val_df, "input", MAX_INPUT_LENGTH
)
X_test_input = prepare_sequences(
    test_df, "input", MAX_INPUT_LENGTH
)

# Combined input + output data for RQ2
X_train_both = prepare_sequences(
    train_df, "both", MAX_OUTPUT_LENGTH
)
X_val_both = prepare_sequences(
    val_df, "both", MAX_OUTPUT_LENGTH
)
X_test_both = prepare_sequences(
    test_df, "both", MAX_OUTPUT_LENGTH
)

# Numerical labels
y_train = train_df["label"].tolist()
y_val = val_df["label"].tolist()
y_test = test_df["label"].tolist()

print("\n=== Prepared Model Data ===")
print("Output training:", len(X_train_output), "x", len(X_train_output[0]))
print("Input training:", len(X_train_input), "x", len(X_train_input[0]))
print("Combined training:", len(X_train_both), "x", len(X_train_both[0]))
print("Training labels:", len(y_train))
print("Validation labels:", len(y_val))
print("Testing labels:", len(y_test))
