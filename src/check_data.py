
# THIS IS SIMPLY A CHECK FOR THE ORIGINAL CSV FILE. NOT IMPORTANT FOR PREPROCESSING.

import pandas as pd
from pathlib import Path

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "llm_responses_dataset.csv"

EXPECTED_MODELS = {"GPT", "Claude", "Gemini"}
EXPECTED_COLUMNS = {
    "LLM_name",
    "prompt_id",
    "category",
    "LLM_Input",
    "LLM_output"
}
COLUMN_ALIASES = {
    column.casefold(): column
    for column in EXPECTED_COLUMNS
}


def main():
    print("=== Dataset Check ===\n")

    # --------------------------------------------------
    # 1. Try loading the CSV
    # --------------------------------------------------
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception as e:
        print("FAILED: pandas could not read the CSV.")
        print(e)
        return

    df.rename(
        columns=lambda column: COLUMN_ALIASES.get(
            str(column).casefold(),
            column
        ),
        inplace=True
    )

    print("PASS: pandas successfully loaded the CSV.")

    # --------------------------------------------------
    # 2. Check columns
    # --------------------------------------------------
    missing_columns = EXPECTED_COLUMNS - set(df.columns)

    if missing_columns:
        print(f"FAIL: Missing columns: {missing_columns}")
        return
    else:
        print("PASS: All expected columns are present.")

    # --------------------------------------------------
    # 3. Check total number of rows
    # --------------------------------------------------
    print(f"\nTotal responses: {len(df)}")

    # --------------------------------------------------
    # 4. Check responses per model
    # --------------------------------------------------
    print("\nResponses per model:")
    print(df["LLM_name"].value_counts())

    unexpected_models = set(df["LLM_name"].unique()) - EXPECTED_MODELS

    if unexpected_models:
        print(f"\nWARNING: Unexpected model names: {unexpected_models}")
    else:
        print("\nPASS: Model names look correct.")

    # --------------------------------------------------
    # 5. Check missing values
    # --------------------------------------------------
    print("\nMissing values:")
    print(df.isnull().sum())

    if df.isnull().any().any():
        print("\nWARNING: Dataset contains missing values.")
    else:
        print("\nPASS: No missing values.")

    # --------------------------------------------------
    # 6. Check duplicate model/prompt combinations
    # --------------------------------------------------
    duplicates = df.duplicated(
        subset=["LLM_name", "prompt_id"],
        keep=False
    )

    if duplicates.any():
        print("\nWARNING: Duplicate model/prompt combinations found:")
        print(
            df.loc[
                duplicates,
                ["LLM_name", "prompt_id"]
            ].sort_values("prompt_id")
        )
    else:
        print("\nPASS: No duplicate model/prompt combinations.")

    # --------------------------------------------------
    # 7. Check every prompt has all 3 models
    # --------------------------------------------------
    print("\nChecking models for each prompt...")

    problems = []

    for prompt_id, group in df.groupby("prompt_id"):
        models = set(group["LLM_name"])

        if models != EXPECTED_MODELS:
            problems.append((prompt_id, models))

    if problems:
        print("WARNING: Some prompts do not have all three models:")

        for prompt_id, models in problems:
            missing = EXPECTED_MODELS - models
            print(
                f"  Prompt {prompt_id}: "
                f"missing {', '.join(sorted(missing))}"
            )
    else:
        print("PASS: Every prompt has GPT, Claude, and Gemini.")

    # --------------------------------------------------
    # 8. Verify same input was used for each model
    # --------------------------------------------------
    inconsistent_inputs = []

    for prompt_id, group in df.groupby("prompt_id"):

        if group["LLM_Input"].nunique() != 1:
            inconsistent_inputs.append(prompt_id)

    if inconsistent_inputs:
        print(
            "\nWARNING: Different LLM_Input text found for "
            f"prompt IDs: {inconsistent_inputs}"
        )
    else:
        print(
            "\nPASS: All models received the same input "
            "for each prompt."
        )

    # --------------------------------------------------
    # 9. Make sure responses aren't empty
    # --------------------------------------------------
    empty_outputs = (
        df["LLM_output"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    if empty_outputs.any():
        print("\nWARNING: Empty LLM outputs found:")
        print(
            df.loc[
                empty_outputs,
                ["LLM_name", "prompt_id"]
            ]
        )
    else:
        print("\nPASS: All LLM outputs contain text.")

    # --------------------------------------------------
    # Final summary
    # --------------------------------------------------
    print("\n==============================")
    print("Dataset check complete.")
    print("==============================")


if __name__ == "__main__":
    main()