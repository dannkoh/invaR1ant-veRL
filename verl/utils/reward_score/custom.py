import re
from typing import Dict, Optional, Tuple, Any

# Z3 Python API imports for in-process SMT solving
from z3 import Solver, parse_smt2_string, Not, And, Z3Exception, sat

# Pre-compiled regex patterns for extract_solution to avoid repeated compilation
_PERFECT_RE = re.compile(
    r"\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>\s*$",
    re.DOTALL,
)
_BASIC_RE = re.compile(
    r"\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>",
    re.DOTALL,
)

# Constants for reward score tiers
SCORE_CORRECT_PERFECT = 1.0
SCORE_CORRECT_WITH_TRAILING = 0.95
SCORE_INCORRECT_PERFECT = 0.1
SCORE_INCORRECT_WITH_TRAILING = 0.05
SCORE_MALFORMED = 0.0


def extract_solution(response: str) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Extract solution content and evaluate formatting quality.

    Args:
        response: The full model response string containing <think> and <answer> tags.

    Returns:
        A tuple of (solution_text, formatting_score, chain_of_thought):
        - formatting_score: 1.0 for perfect (no trailing text),
                             0.5 for correct tags with trailing text,
                             0.0 if tags are missing.
    """
    # Try perfect formatting match (no trailing text)
    perfect_match = _PERFECT_RE.search(response)
    if perfect_match:
        chain_of_thought = perfect_match.group(1).strip()
        solution = perfect_match.group(2).strip()
        return solution, 1.0, chain_of_thought

    # Try basic match (allows trailing text after answer)
    basic_match = _BASIC_RE.search(response)
    if basic_match:
        chain_of_thought = basic_match.group(1).strip()
        solution = basic_match.group(2).strip()
        return solution, 0.5, chain_of_thought

    # No valid <think>/<answer> structure found
    return None, 0.0, None


def check_logical_equivalence(
    original_assertions: str,
    generated_assertions: str,
    constants: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check logical equivalence of two sets of SMT constraints using Z3's incremental API.

    Performs two checks on a single Solver instance with push/pop:
      1. A ⇒ B  (unsat if A ∧ ¬B holds no model)
      2. B ⇒ A  (unsat if B ∧ ¬A holds no model)

    Args:
        original_assertions: full SMT-LIB text including declarations and asserts for A.
        generated_assertions: only assert lines from the LLM output (B).
        constants: optional declaration lines to prepend to generated side.

    Returns:
        A dict with 'result': bool and 'reason': str describing equivalence.
    """
    # Normalize and strip inputs
    orig_smt = original_assertions.strip() if original_assertions else ""
    gen_smt = generated_assertions.strip() if generated_assertions else ""

    # Trivial equivalence cases
    if not orig_smt and not gen_smt:
        return {"result": True, "reason": "Both constraints empty."}
    if not orig_smt or not gen_smt:
        return {"result": False, "reason": "One side is empty but not the other."}

    # Prepend declarations for the generated side, if provided
    if constants:
        gen_smt = constants.strip() + "\n" + gen_smt
        orig_smt = constants.strip() + "\n" + orig_smt

    try:
        # Parse SMT-LIB into Python lists of BoolRef constraints
        orig_constraints = parse_smt2_string(orig_smt)
        gen_constraints = parse_smt2_string(gen_smt)
    except Z3Exception as e:
        return {"result": False, "reason": f"Z3 parse error: {e}"}

    # Use a single Solver for both checks to reuse learned clauses
    s = Solver()

    # Check A ⇒ B: unsat if A ∧ ¬B is contradictory
    s.push()
    s.add(*orig_constraints)
    s.add(Not(And(*gen_constraints)))
    if s.check() == sat:
        s.pop()
        return {"result": False, "reason": "Original does not imply generated."}
    s.pop()

    # Check B ⇒ A: unsat if B ∧ ¬A is contradictory
    s.push()
    s.add(*gen_constraints)
    s.add(Not(And(*orig_constraints)))
    if s.check() == sat:
        s.pop()
        return {"result": False, "reason": "Generated does not imply original."}
    s.pop()

    # Both implications hold => equivalent
    return {"result": True, "reason": "Constraints are equivalent."}


def compute_score(
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any]
) -> float:
    """
    Compute a scalar reward between 0.0 and 1.0 based on correctness and formatting.

    Steps:
     1. Extract solution using extract_solution().
     2. If extraction fails, return SCORE_MALFORMED.
     3. Perform logical equivalence check of solution vs. ground_truth using Z3.
     4. Map equivalence + formatting result to a final score.
    """
    # print(f"\n\n{'#'*30}DEBUG{'#'*30}\n\n{solution_str}\n{'-'*60}")
    # Parse out solution and formatting
    generated_solution, formatting_quality, cot = extract_solution(solution_str)
    if generated_solution is None:
        # Malformed output (missing tags)
        return SCORE_MALFORMED

    # Get declarations if available
    constants = extra_info.get("answer_constants") if extra_info else None

    # Check logic equivalence
    result = check_logical_equivalence(
        original_assertions=ground_truth,
        generated_assertions=generated_solution,
        constants=constants
    )
    equivalent = result.get("result", False)

    # if cot:
    #     print(f"\nChain-of-thought:\n{cot}", flush=True)
    # print(f"\nSolution:\n{generated_solution}", flush=True)
    # print(f"\nResult: {result}", flush=True)
    # print(f"Formatting quality: {formatting_quality}", flush=True)
    # print("#"*60, flush=True)

    # Final scoring logic
    if equivalent:
        # Correct answer
        return SCORE_CORRECT_PERFECT if formatting_quality == 1.0 else SCORE_CORRECT_WITH_TRAILING
    else:
        # Incorrect answer but well-formed tags
        return (
            SCORE_INCORRECT_PERFECT
            if formatting_quality == 1.0
            else SCORE_INCORRECT_WITH_TRAILING
        )