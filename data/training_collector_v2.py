import argparse
import hashlib
import random
import re
import uuid
from itertools import combinations
from pathlib import Path
from tqdm import tqdm
from functools import lru_cache

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
    Rename variable names in a list of texts (e.g. example solution texts and answer solution text)
    based on variable names extracted from constants_text.

    Returns a tuple: (renamed_texts, renamed_constants_text, mapping)
    """
    mapping = {}
    if constants_text is not None:
        var_names = list(extract_variable_names(constants_text))
        rng = random.Random(seed)
        # Use a diverse set of candidate prefixes.
        prefixes = ["in", "var", "tmp", "aux", "res", "out", "arg", "param", "local", "global"]
        prefixes.extend(list("abcdefghijklmnopqrstuvwxyz"))
        new_names = [f"{rng.choice(prefixes)}_{i+1}" for i in range(len(var_names))]
        mapping = dict(zip(var_names, new_names))

    if mapping:
        pattern = re.compile(r"\b(" + "|".join(re.escape(key) for key in mapping.keys()) + r")\b")
        def replace_vars(text: str) -> str:
            if text is None:
                return None
            return pattern.sub(lambda m: mapping[m.group(0)], text)
    else:
        def replace_vars(text: str) -> str:
            return text

    renamed_texts = [replace_vars(t) for t in texts]
    renamed_constants_text = replace_vars(constants_text) if constants_text is not None else None
    return renamed_texts, renamed_constants_text, mapping


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


def generate_index_pairs_generator(max_n: int, min_examples: int, max_examples: int, difference: int):
    """
    Generator version that yields (combo, answer_index) pairs without storing all of them in memory.
    """
    candidate_indices = list(range(1, max_n + 1))
    for k in range(min_examples, max_examples + 1):
        for combo in combinations(candidate_indices, k):
            last_example = combo[-1]  # combinations yield sorted tuples
            for d in range(1, difference + 1):
                answer_index = last_example + d
                if answer_index <= max_n:
                    yield combo, answer_index


def generate_question_entries_for_problem(
    files_dict: dict[int, Path],
    problem: str,
    global_seed: int,
    instruct: bool,
    min_examples: int = 1,
    max_examples: int = 5,
    difference: int = 3,
    max_n: int = 30
) -> list[dict]:
    """
    For a single problem, generate question instances by:
      1) Using a generator for index pairs to avoid holding all combinations in memory.
      2) Caching parsed file contents.
    """
    # Pre-cache parsed file data to avoid re-reading the same file multiple times.
    parsed_files = {i: parse_smt2_file(path) for i, path in files_dict.items()}

    question_entries = []

    for combo, answer_index in generate_index_pairs_generator(max_n, min_examples, max_examples, difference):
        # Deterministic seed for per-(combo, answer_index)
        seed_input = f"{problem}_{combo}_{answer_index}_{global_seed}"
        local_seed = int(hashlib.md5(seed_input.encode()).hexdigest(), 16) % (2**32)

        # Get answer data from cache if it exists.
        answer_parsed = parsed_files.get(answer_index, {"constants": None, "solution": None})

        # Retrieve each example from cache.
        examples = []
        for i in combo:
            parsed = parsed_files.get(i, {"constants": None, "solution": None})
            examples.append({"index": i, "solution": parsed["solution"]})

        # Combine example texts and answer text for variable renaming.
        all_texts = [ex["solution"] for ex in examples] + [answer_parsed["solution"]]
        rename_seed_input = f"{problem}_{combo}_{answer_index}_rename_{global_seed}"
        rename_seed = int(hashlib.md5(rename_seed_input.encode()).hexdigest(), 16) % (2**32)

        renamed_texts, renamed_answer_constants, mapping = rename_variables_in_texts(
            all_texts,
            answer_parsed["constants"],
            rename_seed
        )

        renamed_examples = [
            {"index": ex["index"], "solution": renamed_texts[idx]}
            for idx, ex in enumerate(examples)
        ]
        renamed_answer_solution = renamed_texts[-1]

        examples_str = "\n".join(
            f"N={ex['index']}: {ex['solution']}" for ex in renamed_examples
        )
        question_text = make_prefix("Qwen", examples_str, answer_index, instruct)

        entry = {
            "problem": problem,
            "example_indices": list(combo),
            "examples": renamed_examples,
            "question": question_text,
            "answer_index": answer_index,
            "answer_constants": renamed_answer_constants,
            "answer_solution": renamed_answer_solution,
            "variable_mapping": mapping,
        }
        question_entries.append(entry)

    return question_entries


def collect_smt2_dataframe(base_dir: str, global_seed: int, instruct: bool) -> pd.DataFrame:
    """
    Collect and organize SMT2 data from the specified directory into a DataFrame.
    Processes each problem incrementally to reduce memory usage.
    """
    problems_files = group_smt2_files(base_dir)
    all_entries = []
    for problem, files_dict in tqdm(problems_files.items()):
        # Adjust these parameters as needed.
        entries = generate_question_entries_for_problem(
            files_dict,
            problem,
            global_seed,
            instruct,
            min_examples=3,
            max_examples=5,
            difference=3,
            max_n=30
        )
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
    parser.add_argument("--instruct", action="store_true", help="Whether to follow Qwen instruction format.")
    return parser.parse_args()


def main() -> None:
    """
    Collect SMT2 data, generate question instances, create a DataFrame, and write it to a Parquet file.
    """
    args = parse_arguments()
    df = collect_smt2_dataframe(args.base_dir, args.seed, args.instruct)
    df.to_parquet(args.output)


if __name__ == "__main__":
    main()
