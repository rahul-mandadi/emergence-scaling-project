import json
import os
import sys

def verify_bbh_tasks():
    """Verify selected tasks exist and have enough examples"""
    print("Verifying BIG-Bench Hard tasks...")
    tasks = ['date_understanding', 'boolean_expressions', 'geometric_shapes',
         'tracking_shuffled_objects_five_objects', 'word_sorting']

    # This path is relative to your project root
    base_path = 'data/bbh/bbh/' 

    if not os.path.exists(base_path):
        print(f"ERROR: Directory not found: {base_path}")
        print("Did you move the 'BIG-Bench-Hard' repo into the 'data/bbh' folder correctly?")
        sys.exit(1)

    all_verified = True
    for task in tasks:
        filepath = f'{base_path}{task}.json'

        if not os.path.exists(filepath):
            print(f"  ✗ ERROR: Task file NOT FOUND: {filepath}")
            all_verified = False
            continue

        with open(filepath) as f:
            data = json.load(f)

        n_examples = len(data['examples'])
        # Need 10 for testing + 3 for few-shot = 13
        if n_examples < 13:
            print(f"  ✗ ERROR: Task '{task}' has only {n_examples} examples (need 13+)")
            all_verified = False
        else:
            print(f"  ✓ {task}: OK ({n_examples} examples available)")

    if all_verified:
        print("\n✅ All tasks verified! Ready to collect data.")
    else:
        print("\n❌ Errors found. Please check file paths and task names.")

if __name__ == "__main__":
    verify_bbh_tasks()