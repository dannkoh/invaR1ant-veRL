import re
from typing import Dict, Optional, Tuple, Any
from functools import lru_cache

# Z3 Python API imports for in-process SMT solving
from z3 import Solver, parse_smt2_string, Not, And, Z3Exception, sat

# Constants for reward score tiers
SCORE_CORRECT_PERFECT = 1.0
SCORE_CORRECT_WITH_TRAILING = 0.95
SCORE_INCORRECT_PERFECT = 0.1
SCORE_INCORRECT_WITH_TRAILING = 0.05
SCORE_MALFORMED = 0.0

# Cache parsed SMT constraints to avoid repeated parsing
@lru_cache(maxsize=128)
def _parse_smt(smt_text: str):
    """
    Parse SMT-LIB text into Z3 BoolRef list, cached for reuse.
    """
    return parse_smt2_string(smt_text)


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
    # locate the end of the chain-of-thought section
    think_end = response.rfind("</think>")
    if think_end == -1:
        return None, 0.0, None

    # locate the start of the answer section
    answer_start_tag = "<answer>"
    answer_start = response.find(answer_start_tag, think_end)
    if answer_start == -1:
        return None, 0.0, None
    answer_start += len(answer_start_tag)

    # locate the end of the answer section
    answer_end_tag = "</answer>"
    answer_end = response.find(answer_end_tag, answer_start)
    if answer_end == -1:
        return None, 0.0, None

    # extract the components
    chain_of_thought = response[:think_end].strip()
    solution = response[answer_start:answer_end].strip()
    trailing = response[answer_end + len(answer_end_tag):].strip()

    # determine formatting score
    formatting_score = 1.0 if not trailing else 0.5
    return solution, formatting_score, chain_of_thought


def check_logical_equivalence(
    original_assertions: str,
    generated_assertions: str,
    constants: Optional[str] = None
) -> Dict[str, Any]:
    """
    Check logical equivalence of two sets of SMT constraints using Z3's incremental API.

    Performs two checks on a single Solver instance with push/pop:
      1. A ⇒ B  (unsat if A ∧ ¬B is satisfiable)
      2. B ⇒ A  (unsat if B ∧ ¬A is satisfiable)

    Args:
        original_assertions: full SMT-LIB text including asserts for A.
        generated_assertions: SMT-LIB asserts for B.
        constants: optional SMT-LIB declarations to prepend.

    Returns:
        A dict with 'result': bool and 'reason': str describing equivalence.
    """
    # Normalize and strip inputs
    orig_smt = original_assertions.strip() if original_assertions else ""
    gen_body = generated_assertions.strip() if generated_assertions else ""

    # Trivial equivalence cases
    if not orig_smt and not gen_body:
        return {"result": True, "reason": "Both constraints empty."}
    if not orig_smt or not gen_body:
        return {"result": False, "reason": "One side is empty but not the other."}

    # Prepend declarations if provided
    if constants:
        decls = constants.strip()
        gen_smt = decls + "\n" + gen_body
        orig_smt = decls + "\n" + orig_smt
    else:
        gen_smt = gen_body

    try:
        # Parse SMT-LIB into lists of constraints, cached
        orig_constraints = _parse_smt(orig_smt)
        gen_constraints = _parse_smt(gen_smt)
    except Z3Exception as e:
        return {"result": False, "reason": f"Z3 parse error: {e}"}

    # Use a single Solver for both checks to reuse learned clauses
    solver = Solver()

    # Check A ⇒ B: A ∧ ¬B should be unsatisfiable
    solver.push()
    solver.add(*orig_constraints)
    solver.add(Not(And(*gen_constraints)))
    if solver.check() == sat:
        solver.pop()
        return {"result": False, "reason": "Original does not imply generated."}
    solver.pop()

    # Check B ⇒ A: B ∧ ¬A should be unsatisfiable
    solver.push()
    solver.add(*gen_constraints)
    solver.add(Not(And(*orig_constraints)))
    if solver.check() == sat:
        solver.pop()
        return {"result": False, "reason": "Generated does not imply original."}
    solver.pop()

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
    # Extract solution and formatting quality
    generated_solution, formatting_quality, chain_of_thought = extract_solution(solution_str)
    if generated_solution is None:
        # Malformed output (missing tags)
        return SCORE_MALFORMED

    # Retrieve declarations if available
    constants = extra_info.get("answer_constants") if extra_info else None

    # Perform logical equivalence check
    result = check_logical_equivalence(
        original_assertions=ground_truth,
        generated_assertions=generated_solution,
        constants=constants
    )
    equivalent = result.get("result", False)

    # Map to final score
    if equivalent:
        return SCORE_CORRECT_PERFECT if formatting_quality == 1.0 else SCORE_CORRECT_WITH_TRAILING
    else:
        return (
            SCORE_INCORRECT_PERFECT
            if formatting_quality == 1.0
            else SCORE_INCORRECT_WITH_TRAILING
        )