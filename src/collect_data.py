import csv
import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROMPTS_FILE = DATA_DIR / "prompts.csv"
RESPONSES_FILE = DATA_DIR / "llm_responses_dataset.csv"


def load_prompts():
    """Load all prompts from prompts.csv."""
    prompts = {}

    with open(PROMPTS_FILE, "r", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        for row in reader:
            prompt_id = row["prompt_id"]

            prompts[prompt_id] = {
                "category": row["category"],
                "prompt": row["prompt"]
            }

    return prompts


def get_multiline_response():
    """
    Allow the user to paste a multi-line LLM response.
    Type END on its own line when finished.
    """
    print("\nPaste the LLM response below.")
    print("When finished, type END on its own line.\n")

    lines = []

    while True:
        line = input()

        if line == "END":
            break

        lines.append(line)

    return "\n".join(lines)


def save_response(llm_name, prompt_id, category, prompt, response):
    """Append one LLM response to llm_responses.csv."""

    file_exists = os.path.exists(RESPONSES_FILE)

    with open(RESPONSES_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # Add column names if we're creating the dataset
        if not file_exists:
            writer.writerow([
                "LLM_name",
                "prompt_id",
                "category",
                "LLM_Input",
                "LLM_output"
            ])

        writer.writerow([
            llm_name,
            prompt_id,
            category,
            prompt,
            response
        ])


def main():

    prompts = load_prompts()

    print("=== LLM Response Collector ===")

    while True:

        # Ask which LLM generated the response
        llm_name = input(
            "\nEnter LLM name (GPT, Claude, Gemini): "
        ).strip()

        # Ask which prompt was used
        prompt_id = input("Enter prompt ID: ").strip()

        if prompt_id not in prompts:
            print("Invalid prompt ID.")
            continue

        category = prompts[prompt_id]["category"]
        prompt = prompts[prompt_id]["prompt"]

        print("\nCategory:", category)
        print("Prompt:", prompt)

        # Collect the complete response
        response = get_multiline_response()

        # Make sure something was actually entered
        if not response.strip():
            print("No response entered. Nothing was saved.")
            continue

        # Save response to dataset
        save_response(
            llm_name,
            prompt_id,
            category,
            prompt,
            response
        )

        print("\nResponse saved successfully.")

        again = input("Add another response? (y/n): ").strip().lower()

        if again != "y":
            break

    print("\nFinished collecting responses.")


if __name__ == "__main__":
    main()