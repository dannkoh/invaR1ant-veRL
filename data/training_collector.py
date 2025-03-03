import argparse
import hashlib
import itertools
import random
import re
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


def extract_variable_names_in_order(constants_text: str) -> list[str]:
    """
    Extract variable names from constants text using the pattern in declare-const lines,
    preserving the order of first appearance.
    """
    var_names = []
    for match in re.finditer(r"\(declare-const\s+(\S+)", constants_text):
        var = match.group(1)
        if var not in var_names:
            var_names.append(var)
    return var_names


def canonical_rename_variables_in_texts(
    texts: list[str], constants_text: str, canonical_prefix: str
) -> (list[str], str, dict):
    """
    Generate a canonical mapping based on the answer's constants text.
    Every variable encountered (in order of first appearance) is mapped to a canonical name
    using the given canonical_prefix (e.g. if canonical_prefix="qwerty", then names like "qwerty0", "qwerty1", etc.).
    The mapping is then applied uniformly to all texts and to the constants text.
    If constants_text is None, returns texts unchanged.
    """
    if constants_text is None:
        return texts, constants_text, {}

    # Extract variable names in order from the constants text.
    var_names = extract_variable_names_in_order(constants_text)
    mapping = {var: f"{canonical_prefix}{idx}" for idx, var in enumerate(var_names)}

    # Build a regex to match any variable name (using word boundaries)
    pattern = re.compile(r"\b(" + "|".join(re.escape(var) for var in mapping.keys()) + r")\b")

    def replace_vars(text: str) -> str:
        return pattern.sub(lambda m: mapping[m.group(0)], text)

    renamed_texts = [replace_vars(t) if t is not None else t for t in texts]
    renamed_constants_text = replace_vars(constants_text)
    return renamed_texts, renamed_constants_text, mapping


# Provided make_prefix function (unchanged).
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
    For example: <answer>(assert (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (not ( = in0 64)) (not ( = in0 35))) (not ( = in0 36)))  ( =  in0 37)) (not ( = in1 64))) (not ( = in1 35))) (not ( = in1 36)))  ( =  in1 37)) (not ( = in2 64))) (not ( = in2 35))) (not ( = in2 36)))  ( =  in2 37)) (not ( = in3 64))) (not ( = in3 35))) (not ( = in3 36)))  ( =  in3 37)) (not ( = in4 64))) (not ( = in4 35))) (not ( = in4 36)))  ( =  in4 37)) (not ( = in5 64))) (not ( = in5 35))) (not ( = in5 36)))  ( =  in5 37)) (not ( = in6 64))) (not ( = in6 35))) (not ( = in6 36)))  ( =  in6 37)) (not ( = in7 64))) (not ( = in7 35))) (not ( = in7 36)))  ( =  in7 37)) (not ( = in8 64))) (not ( = in8 35))) (not ( = in8 36)))  ( =  in8 37)) (not ( = in9 64))) (not ( = in9 35))) (not ( = in9 36)))  ( =  in9 37)))</answer>.
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
For example: <answer>(assert (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (and (not ( = in0 64)) (not ( = in0 35))) (not ( = in0 36)))  ( =  in0 37)) (not ( = in1 64))) (not ( = in1 35))) (not ( = in1 36)))  ( =  in1 37)) (not ( = in2 64))) (not ( = in2 35))) (not ( = in2 36)))  ( =  in2 37)) (not ( = in3 64))) (not ( = in3 35))) (not ( = in3 36)))  ( =  in3 37)) (not ( = in4 64))) (not ( = in4 35))) (not ( = in4 36)))  ( =  in4 37)) (not ( = in5 64))) (not ( = in5 35))) (not ( = in5 36)))  ( =  in5 37)) (not ( = in6 64))) (not ( = in6 35))) (not ( = in6 36)))  ( =  in6 37)) (not ( = in7 64))) (not ( = in7 35))) (not ( = in7 36)))  ( =  in7 37)) (not ( = in8 64))) (not ( = in8 35))) (not ( = in8 36)))  ( =  in8 37)) (not ( = in9 64))) (not ( = in9 35))) (not ( = in9 36)))  ( =  in9 37)))</answer>.
Here are the known constraints:
{examples}
What is the constraint for N={N_question}?
Assistant: Let me solve this step by step.
<think>"""


def generate_question_entries_for_problem(
    files_dict: dict[int, Path],
    problem: str,
    min_examples: int,
    max_examples: int,
    answer_range: range,
    global_seed: int,
    model: str = "Qwen"
) -> list[dict]:
    """
    For a single problem, generate one question instance for every combination of example indices
    (from min_examples to max_examples, chosen from indices 1..10) and for each answer index in answer_range (11..30).

    The answer's constants text is used to build a canonical mapping for variable renaming,
    which is applied uniformly to all example solution texts, the answer solution, and the constants.
    Both instruct and non-instruct prompts are generated.
    """
    candidate_example_indices = list(range(1, 11))
    question_entries = []

    for k in range(min_examples, max_examples + 1):
        for combo in itertools.combinations(candidate_example_indices, k):
            for answer_index in answer_range:
                # Parse the answer file (or placeholder)
                if answer_index in files_dict:
                    answer_parsed = parse_smt2_file(files_dict[answer_index])
                else:
                    answer_parsed = {"constants": None, "solution": None}

                # Parse example files (or placeholders)
                examples = []
                for i in sorted(combo):
                    if i in files_dict:
                        parsed = parse_smt2_file(files_dict[i])
                    else:
                        parsed = {"constants": None, "solution": None}
                    examples.append({"index": i, "solution": parsed["solution"]})

                # Generate a random (but reproducible) prefix for this instance.
                seed_input = f"{problem}_{combo}_{answer_index}_{global_seed}"
                instance_seed = int(hashlib.md5(seed_input.encode()).hexdigest(), 16) % (2**32)
                rng = random.Random(instance_seed)
                alphabet = "abcdefghijklmnopqrstuvwxyz"
                prefix_length = 3
                random_prefix = "".join(rng.choice(alphabet) for _ in range(prefix_length))

                # Apply canonical renaming with the random prefix
                all_texts = [ex["solution"] for ex in examples] + [answer_parsed["solution"]]
                renamed_texts, renamed_constants, mapping = canonical_rename_variables_in_texts(
                    all_texts, answer_parsed["constants"], random_prefix
                )

                renamed_examples = []
                for idx, ex in enumerate(examples):
                    renamed_examples.append({"index": ex["index"], "solution": renamed_texts[idx]})
                renamed_answer_solution = renamed_texts[-1]

                examples_str = "\n".join(f"N={ex['index']}: {ex['solution']}" for ex in renamed_examples)
                prompt_instruct = make_prefix(model, examples_str, answer_index, True)
                prompt_plain = make_prefix(model, examples_str, answer_index, False)

                entry = {
                    "problem": problem,
                    "example_indices": list(combo),
                    "examples": renamed_examples,
                    "question_instruct": prompt_instruct,
                    "question_plain": prompt_plain,
                    "answer_index": answer_index,
                    "answer_constants": renamed_constants,
                    "answer_solution": renamed_answer_solution,
                    "variable_mapping": mapping,
                    "instance_prefix": random_prefix,  # for debugging/inspection
                }
                question_entries.append(entry)
    return question_entries


def collect_smt2_dataframe(
    base_dir: str,
    min_examples: int,
    max_examples: int,
    answer_range: range,
    global_seed: int,
    model: str,
) -> pd.DataFrame:
    """
    Collect SMT2 data from the specified directory, group by problem, and generate a DataFrame of question instances.
    """
    problems_files = group_smt2_files(base_dir)
    all_entries = []
    for problem, files_dict in problems_files.items():
        entries = generate_question_entries_for_problem(
            files_dict, problem, min_examples, max_examples, answer_range, global_seed, model
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
    parser.add_argument("--seed", type=int, default=69420, help="Global random seed for reproducibility.")
    parser.add_argument("--min_examples", type=int, default=3, help="Minimum number of examples per instance.")
    parser.add_argument("--max_examples", type=int, default=10, help="Maximum number of examples per instance.")
    parser.add_argument("--answer_start", type=int, default=11, help="Starting index for answers.")
    parser.add_argument("--answer_end", type=int, default=30, help="Ending index for answers (inclusive).")
    parser.add_argument("--model", type=str, default="Qwen", help="Model name to use in the prompt.")
    return parser.parse_args()


def main() -> None:
    """
    Collect SMT2 data from the specified directory, generate question instances,
    create a DataFrame, and write it to a Parquet file.
    """
    args = parse_arguments()
    answer_range = range(args.answer_start, args.answer_end + 1)
    df = collect_smt2_dataframe(
        args.base_dir,
        args.min_examples,
        args.max_examples,
        answer_range,
        args.seed,
        args.model,
    )
    df.to_parquet(args.output)
    print(f"Data saved to {args.output}")


if __name__ == "__main__":
    main()
