import subprocess
import re


def compute_score(solution_str: str, ground_truth: str, extra_info: dict[str,str], is_instruct: bool = False) -> float:
    """
    Compute the score for a given solution and ground truth by checking
    the logical equivalence of their SMT constraints.

    Args:
        solution_str (str): The solution string.
        ground_truth (dict): The ground truth data.
        extra_info (dict): Extra information.

    Returns:
        float: The score. 1 for correct, 0.1 for incorrect and 0 for poorly formatted.

    """
    # Check if solution string has a single </think>, <answer> and </answer> tag
    generated_solution = extract_solution(solution_str, is_instruct)
    if generated_solution is None:
        return 0

    answer_constants = extra_info.get("answer_constants", None)

    result = check_logical_equivalence(
        generated_assertions=generated_solution, original_assertions=ground_truth, constants=answer_constants
    )

    print(f"\n\nResult: {result}", flush=True)
    print("#"*60, flush=True)


    if (not result["result"] and result["reason"] == "Not Equivalent") or (not result["result"] and result["reason"] == "Incomplete Z3 output."):
        return 0.1
    elif not result["result"]:
        return 0
    return 1


def check_logical_equivalence(original_assertions, generated_assertions, constants):
    """
    Check the logical equivalence between the original and generated constraints
    using Z3 by piping SMT-LIB content directly to its standard input.

    Args:
        original_assertions (str): The original assertions.
        generated_assertions (str): The generated assertions.
        constants (str): The constants from original assertions.

    """
    accepted = ["None", "(assert True)", "(assert False)", "", None]
    if original_assertions in accepted:
        original_assertions = None
    if generated_assertions in accepted:
        generated_assertions = None

    if not original_assertions and not generated_assertions:
        return {"result": True, "reason": "Both Empty"}

    if not original_assertions or not generated_assertions:
        return {"result": False, "reason": "Not Equivalent"}

    smt_content = f"""; Combined SMT for checking equivalence
; Original constants:
{constants}

; Original constraints (A):
(push)
{original_assertions}
(pop)

; Generated constraints (B):
(push)
{generated_assertions}
(pop)

; Check 1: Does A imply B? (i.e. is A ∧ not(B) unsatisfiable?)
(push)
{original_assertions}
(assert (not
{parse_raw_constraints(generated_assertions)}
))
(check-sat)
(pop)

; Check 2: Does B imply A? (i.e. is B ∧ not(A) unsatisfiable?)
(push)
{generated_assertions}
(assert (not
{parse_raw_constraints(original_assertions)}
))
(check-sat)
(pop)
"""
    # Check the size of the SMT content.
    smt_size = len(smt_content.encode("utf-8"))
    if smt_size > 2_000_000:
        return {"result": False, "reason": "Constraints generated are too large."}

    # Pipe the SMT content directly to Z3 via standard input.
    try:
        proc = subprocess.run(["z3", "-in"], input=smt_content, capture_output=True, text=True, check=False)
        output = proc.stdout.strip()
        # Gather any error messages.
        results = [line for line in output.splitlines() if line in ("sat", "unsat", "unknown")]
        print(f"\nOriginal Assertions:\n{original_assertions}\n\nGenerated:\n{generated_assertions}\n\n")
        print(f"\nZ3 output:\n{output}\n{"-" * 60}\n")
        if len(results) < 2:
            return {"result": False, "reason": "Incomplete Z3 output."}

        # If either check returns 'sat', then a counterexample exists.
        if results[0] == "sat" or results[1] == "sat":
            return {"result": False, "reason": "Not Equivalent."}

        return {"result": True, "reason": "Constraints are logically equivalent."}

    except Exception as e:
        return {"result": False, "reason": f"Error running Z3: {e}"}


def parse_raw_constraints(constraint):
    """
    Parse raw SMT-LIB2 constraints into a single conjunctive form.
    """
    # Extract all individual assertions
    assertions = [line.strip()[8:-1] for line in constraint.splitlines() if line.startswith("(assert")]
    # Combine into a single conjunctive expression
    return f"(and {' '.join(assertions)})"

def extract_solution(response: str, is_instruct: bool) -> str:
    """
    Extract the solution string from the response.

    This function expects that the response contains:
      - Optionally some manually formatted text before the <think> token.
      - A <think> ... </think> block.
      - An <answer> ... </answer> block.
    The entire response must end immediately after </answer>.

    Args:
      response (str): The full response text.
      is_instruct (bool): If True, expects additional <im_start> tokens.

    Returns:
      str: The solution string from the <answer> block, or None if formatting fails.
    """
    if not is_instruct:
        pattern = (
            r".*<think>\s*(.*?)\s*</think>\s*"   # Capture chain-of-thought block.
            r"<answer>\s*(.*?)\s*</answer>\s*$"    # Capture answer block and require end-of-string.
        )
    else:
        pattern = (
            r".*<im_start>.*?Assistant:\s*.*?"      # Allow any text (including the prompt) before the tokens.
            r"<think>\s*(.*?)\s*</think>\s*"         # Capture chain-of-thought.
            r"<answer>\s*(.*?)\s*</answer>\s*$"       # Capture answer block, with no extra text after.
        )

    # Use fullmatch to ensure the response ends exactly after the answer block.
    match = re.fullmatch(pattern, response, re.DOTALL)
    if match is None:
        return None

    chain_of_thought = match.group(1).strip()
    solution = match.group(2).strip()

    if "<answer>" in solution or "</answer>" in solution:
        return None

    # Debug output showing the first 50 characters of each block.
    print(f"\n\n{'#'*30}DEBUG{'#'*30}\n\n{response}\n{'-'*60}\nChain-of-thought: {chain_of_thought}\nSolution: {solution}\n{'-'*60}", flush=True)

    return solution