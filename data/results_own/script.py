import os
import subprocess
import json

failures = {}
passes = {}

def run_z3(file_path, top_level_dir):
    try:
        result = subprocess.run(['z3', file_path], capture_output=True, text=True)
        if result.returncode != 0:
            log_failure(file_path, result.stderr, top_level_dir)
        else:
            log_pass(top_level_dir)
    except Exception as e:
        log_failure(file_path, str(e), top_level_dir)

def log_failure(file_path, error_message, top_level_dir):
    if top_level_dir not in failures:
        failures[top_level_dir] = []
    failures[top_level_dir].append({
        "file": file_path,
        "error": error_message
    })

def log_pass(top_level_dir):
    if top_level_dir not in passes:
        passes[top_level_dir] = 0
    passes[top_level_dir] += 1

def find_smt2_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.smt2'):
                top_level_dir = os.path.relpath(root, directory).split(os.sep)[0]
                run_z3(os.path.join(root, file), top_level_dir)

if __name__ == "__main__":
    repository_path = os.path.dirname(os.path.abspath(__file__))
    find_smt2_files(repository_path)
    
    with open('z3_failures.json', 'w') as log_file:
        json.dump({"failures": failures, "passes": passes}, log_file, indent=4)