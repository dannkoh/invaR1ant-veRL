import argparse
import hashlib
import random
import re
import uuid
from itertools import combinations
from pathlib import Path

import pandas as pd


def parse_smt2_file(filepath: Path) -> dict[str, str]:
    """
    Parse an SMT2 file to extract constants and solution.
    Lines starting with "(declare-const" are treated as constants.
    Lines starting with "(assert" are treated as part of the solution.
    """
    constants_lines = []
    solution_lines = []
    with filepath.open("r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("(declare-const"):
                constants_lines.append(stripped)
            elif stripped.startswith("(assert"):
                solution_lines.append(stripped)
    return {"constants": "\n".join(constants_lines), "solution": "\n".join(solution_lines)}


def group_smt2_files(base_dir: str) -> dict[str, dict[int, Path]]:
    """
    Walk through the base directory and group SMT2 files by problem name and index.
    Files are expected to be named as: own.<problem>_<N>.smt2.
    """
    base_path = Path(base_dir)
    pattern = re.compile(r"^own\.(?P<problem>.+)_(?P<index>\d+)\.smt2$")
    problems_files: dict[str, dict[int, Path]] = {}

    for filepath in base_path.rglob("*.smt2"):
        match = pattern.match(filepath.name)
        if match:
            problem = match.group("problem")
            index = int(match.group("index"))
            if problem not in problems_files:
                problems_files[problem] = {}
            problems_files[problem][index] = filepath

    return problems_files


def extract_variable_names(constants_text: str) -> set[str]:
    """
    Extract variable names from constants text using the pattern in declare-const lines.
    """
    return set(re.findall(r"\(declare-const\s+(\S+)", constants_text))


def rename_variables_in_texts(texts: list[str], constants_text: str, seed: int) -> (list[str], str, dict):
    """
    (Modified) Removed the renaming/masking of variables.
    Simply returns the original texts, constants text, and an empty mapping.
    """
    return texts, constants_text, {}


def generate_question_entries_for_problem(
    files_dict: dict[int, Path], problem: str, global_seed: int, instruct: bool
) -> list[dict]:
    """
    For a single problem, generate one question instance per possible combination of examples.

    - Examples come from indices 1 to 10.
    - Answer is chosen randomly (but reproducibly) from available files with indices 11 to 30.
    - If a file is missing, its parsed values default to None.
    """
    MIN_EXAMPLES = 3
    MAX_EXAMPLES = 10

    # Always consider indices 1..10 for examples and 11..30 for answers.
    candidate_example_indices = list(range(1, 11))
    candidate_answer_indices = list(range(11, 31))

    question_entries = []
    for k in range(MIN_EXAMPLES, MAX_EXAMPLES + 1):
        for combo in combinations(candidate_example_indices, k):
            # Compute a deterministic local seed for this question instance.
            seed_input = f"{problem}_{combo}_{global_seed}"
            local_seed = int(hashlib.md5(seed_input.encode()).hexdigest(), 16) % (2**32)
            rng = random.Random(local_seed)
            chosen_answer_index = rng.choice(candidate_answer_indices)

            # Parse answer file if exists; otherwise use placeholders.
            if chosen_answer_index in files_dict:
                answer_parsed = parse_smt2_file(files_dict[chosen_answer_index])
            else:
                answer_parsed = {"constants": None, "solution": None}

            # Parse example files (or use placeholders if missing).
            examples = []
            for i in sorted(combo):
                if i in files_dict:
                    parsed = parse_smt2_file(files_dict[i])
                else:
                    parsed = {"constants": None, "solution": None}
                examples.append({"index": i, "solution": parsed["solution"]})

            # Combine texts for renaming: all example solutions and the answer solution.
            all_texts = [ex["solution"] for ex in examples] + [answer_parsed["solution"]]
            rename_seed_input = f"{problem}_{combo}_{chosen_answer_index}_rename_{global_seed}"
            rename_seed = int(hashlib.md5(rename_seed_input.encode()).hexdigest(), 16) % (2**32)
            # The renaming function now simply returns the original texts.
            renamed_texts, renamed_answer_constants, mapping = rename_variables_in_texts(
                all_texts, answer_parsed["constants"], rename_seed
            )

            # Update examples with (unchanged) solution texts.
            renamed_examples = []
            for idx, ex in enumerate(examples):
                renamed_examples.append({"index": ex["index"], "solution": renamed_texts[idx]})
            renamed_answer_solution = renamed_texts[-1]

            # Create a question text using the (unchanged) examples.
            examples_str = "\n".join(f"N={ex['index']}: {ex['solution']}" for ex in renamed_examples)
            question_text = make_prefix("Qwen", examples_str, chosen_answer_index, instruct)

            entry = {
                "problem": problem,
                "example_indices": list(combo),
                "examples": renamed_examples,
                "question": question_text,
                "answer_index": chosen_answer_index,
                "answer_constants": renamed_answer_constants,
                "answer_solution": renamed_answer_solution,
            }
            question_entries.append(entry)
    return question_entries


def make_prefix(model, examples, N_question, instruct) -> str:  # noqa: N803
    if not model.startswith("Qwen"):
        raise NotImplementedError("Only Qwen models are supported.")

    if instruct:
        return f"""<|im_start|>system
    You are a helpful assistant.
    You first think about the reasoning process in your mind and then provide the user with the answer.
    <|im_end|>
    <|im_start|>user
    Your role is to take a known pattern of symbolic constraints that represent the longest execution path of a program 
    and generalize it for any given input size N.
    When you receive an input value N,
    you must generate a canonical SMT-LIB constraint string that adheres to the following rules:
    (assert (op (op (op var_1 var_2)) (op (op var_3 var_4)) (op (op var_5 var_6)) (op var_7 var_8)))
    where op is a logical operator (e.g., 'and', 'or', 'not') and var_i are variables or constants.
    All per-variable constraints must be combined using a top-level (assert (and ...)) clause.
    The output must be in exact, canonical SMT-LIB format without extra commentary in the constraint string.
    Show your work in <think> </think> tags. And return the final SMT-LIB constraint string in <answer> </answer> tags.
    For example: <answer>(assert (and  ( >=  in0 97)  ( <=  in0 122)))</answer>.
    Here are the known constraints:
    {examples}
    What is the constraint for N={N_question}?
    <|im_end|>
    <|im_start|>assistant
    Let me solve this step by step.
    <think>"""

    return f"""A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
User: Your role is to take a known pattern of symbolic constraints that represent the longest execution path of a program 
and generalize it for any given input size N.
When you receive an input value N,
you must generate a canonical SMT-LIB constraint string that adheres to the following rules:
(assert (op (op (op var_1 var_2)) (op (op var_3 var_4)) (op (op var_5 var_6)) (op var_7 var_8)))
where op is a logical operator (e.g., 'and', 'or', 'not') and var_i are variables or constants.
All per-variable constraints must be combined using a top-level (assert (and ...)) clause.
The output must be in exact, canonical SMT-LIB format without extra commentary in the constraint string.
Show your work in <think> </think> tags. And return the final SMT-LIB constraint string in <answer> </answer> tags.
For example: <answer>(assert (and  ( >=  in0 97)  ( <=  in0 122)))</answer>.
Here are the known constraints:
{examples}
What is the constraint for N={N_question}?
Assistant: Let me solve this step by step.
<think>"""


def collect_smt2_dataframe(base_dir: str, global_seed: int, instruct: bool) -> pd.DataFrame:
    """
    Collect and organize SMT2 data from the specified directory into a DataFrame.

    For each problem (grouped by file name), we generate question instances using
    combinations of examples from indices 1 to 10 and selecting an answer from
    indices 11 to 30. Missing files result in None values for constraints and solutions.
    """
    problems_files = group_smt2_files(base_dir)
    all_entries = []
    for problem, files_dict in problems_files.items():
        entries = generate_question_entries_for_problem(files_dict, problem, global_seed, instruct)
        all_entries.extend(entries)
    print(f"Collected {len(all_entries)} question instances.")
    return pd.DataFrame(all_entries)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Collect and organize SMT2 file data into a DataFrame and save as a Parquet file."
    )
    parser.add_argument("base_dir", type=str, help="Base directory to search for SMT2 files.")
    parser.add_argument("output", type=str, help="Output Parquet file.")
    parser.add_argument("--seed", type=int, default=69_420, help="Random seed for reproducibility.")
    parser.add_argument("--instruct", type=bool, default=False, help="Whether to follow Qwen instruction format.")
    return parser.parse_args()


def main() -> None:
    """
    Collect SMT2 data from the specified directory, generate question instances,
    create a DataFrame, and write it to a Parquet file.
    """
    args = parse_arguments()
    df = collect_smt2_dataframe(args.base_dir, args.seed, args.instruct)
    df.to_parquet(args.output)


if __name__ == "__main__":
    main()
