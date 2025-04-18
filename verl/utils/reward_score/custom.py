import subprocess
import re
from typing import Dict, Optional, Tuple, Any

# Constants for reward scores
SCORE_CORRECT_PERFECT = 1.0
SCORE_CORRECT_WITH_TRAILING = 0.95
SCORE_INCORRECT_PERFECT = 0.1
SCORE_INCORRECT_WITH_TRAILING = 0.05
SCORE_MALFORMED = 0.0

def extract_solution(response: str) -> Tuple[Optional[str], float, Optional[str]]:
    """
    Extract solution content and evaluate formatting quality.
    
    Args:
        response: The full model response
        
    Returns:
        Tuple of (extracted_solution, formatting_score, chain_of_thought)
        where formatting_score is:
        - 1.0: Perfect formatting (nothing after </answer>)
        - 0.5: Has correct tags but text after </answer>
        - 0.0: Missing required tags
    """
    # Check for perfect formatting (nothing after </answer>)
    perfect_pattern = r"\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>\s*$"
    perfect_match = re.search(perfect_pattern, response, re.DOTALL)
    
    if perfect_match:
        chain_of_thought = perfect_match.group(1).strip()
        solution = perfect_match.group(2).strip()
        
        return solution, 1.0, chain_of_thought
    
    # Check for basic tag structure but with trailing content
    basic_pattern = r"\s*(.*?)\s*</think>\s*<answer>\s*(.*?)\s*</answer>"
    basic_match = re.search(basic_pattern, response, re.DOTALL)
    
    if basic_match:
        chain_of_thought = basic_match.group(1).strip()
        solution = basic_match.group(2).strip()
        
        return solution, 0.5, chain_of_thought
    
    # Failed to match required pattern
    return None, 0.0, None

def check_logical_equivalence(original_assertions: str, 
                             generated_assertions: str, 
                             constants: Optional[str]) -> Dict[str, Any]:
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
        proc = subprocess.run(["z3", "-in"], input=smt_content, capture_output=True, text=True, check=False, timeout=10)
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

def parse_raw_constraints(constraint: str) -> str:
    """
    Parse raw SMT-LIB2 constraints into a single conjunctive form.
    """
    # Extract all individual assertions
    assertions = [line.strip()[8:-1] for line in constraint.splitlines() if line.startswith("(assert")]
    # Combine into a single conjunctive expression
    return f"(and {' '.join(assertions)})"

def compute_score(solution_str: str, ground_truth: str, 
                 extra_info: Dict[str, str]) -> float:
    """
    Compute the score based on correctness and formatting.
    
    Reward structure:
    - 1.0: Correct answer with perfect formatting
    - 0.95: Correct answer with text after </answer>
    - 0.1: Wrong answer with perfect formatting
    - 0.05: Wrong answer with text after </answer>
    - 0.0: Missing required tags
    
    Args:
        solution_str: The complete solution string
        ground_truth: The expected answer
        extra_info: Additional context information
        
    Returns:
        Float score between 0.0 and 1.0
    """
    # For debugging
    print(f"\n\n{'#'*30}DEBUG{'#'*30}\n\n{solution_str}\n{'-'*60}", flush=True)
    
    # Extract solution and determine formatting quality
    generated_solution, formatting_quality, chain_of_thought = extract_solution(solution_str)
    
    # If extraction failed, return zero
    if generated_solution is None:
        return SCORE_MALFORMED
        
    # Get constants for verification
    answer_constants = extra_info.get("answer_constants", None)
    
    # Check logical equivalence
    result = check_logical_equivalence(
        generated_assertions=generated_solution,
        original_assertions=ground_truth,
        constants=answer_constants
    )
    
    # Print diagnostic information
    if chain_of_thought:
        print(f"\nChain-of-thought:\n{chain_of_thought}", flush=True)
    print(f"\nSolution:\n{generated_solution}", flush=True)
    print(f"\nResult: {result}", flush=True)
    print(f"Formatting quality: {formatting_quality}", flush=True)
    print("#"*60, flush=True)
    
    # Determine final score
    if result["result"]:
        # Correct answer
        return SCORE_CORRECT_PERFECT if formatting_quality == 1.0 else SCORE_CORRECT_WITH_TRAILING
    else:
        # Incorrect answer
        if formatting_quality == 0.0:
            return SCORE_MALFORMED
        return SCORE_INCORRECT_PERFECT if formatting_quality == 1.0 else SCORE_INCORRECT_WITH_TRAILING