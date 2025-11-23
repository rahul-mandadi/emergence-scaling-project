import json
import os
import random

def load_bbh_task(task_name, bbh_dataset_dir):
    """Loads the JSON data for a specific BBH task."""
    filepath = os.path.join(bbh_dataset_dir, f"{task_name}.json")
    try:
        with open(filepath) as f:
            task_data = json.load(f)
        return task_data
    except FileNotFoundError:
        print(f"Error: Task file not found at {filepath}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {filepath}")
        return None

def load_cot_demonstrations(task_name, bbh_cot_dir='data/bbh/cot-prompts'):
    """
    Load the official BBH CoT demonstrations from the cot-prompts directory.
    
    Args:
        task_name: Name of the BBH task
        bbh_cot_dir: Directory containing the CoT prompt files
    
    Returns:
        str: Formatted CoT demonstrations
    """
    cot_file = os.path.join(bbh_cot_dir, f"{task_name}.txt")
    
    try:
        with open(cot_file, 'r') as f:
            content = f.read()
        
        # BBH CoT files contain full Q&A examples with reasoning
        # They're already in the right format, just need to ensure proper spacing
        # Remove any trailing whitespace and ensure consistent formatting
        content = content.strip()
        
        # The BBH CoT files already have the demonstrations in the right format
        # We just return them as-is since they include the full reasoning chains
        return content
        
    except FileNotFoundError:
        print(f"WARNING: CoT prompt file not found at {cot_file}")
        print(f"Falling back to broken CoT implementation for task {task_name}")
        return None
    except Exception as e:
        print(f"Error loading CoT demonstrations: {e}")
        return None

def format_example(example_input, example_target, include_reasoning=False):
    """
    Creates example blocks for few-shot prompting.
    
    Args:
        example_input: The question text
        example_target: The answer (e.g., "(A)" or "True")
        include_reasoning: If True, adds CoT-style reasoning prefix (DEPRECATED - use load_cot_demonstrations instead)
    
    Returns:
        str: Formatted example block
    """
    if include_reasoning:
        # This is the BROKEN implementation - kept for fallback only
        print("WARNING: Using broken CoT implementation. Should use load_cot_demonstrations instead.")
        return f"Q: {example_input}\nA: Let's think step by step. {example_target}"
    else:
        # Regular few-shot: Just Q/A pairs
        return f"Q: {example_input}\nA: {example_target}"

def build_prompt(question, technique, demo_examples, task_name=None):
    """
    Constructs the final prompt string based on the technique.

    Args:
        question (str): The test question to be answered
        technique (str): 'zero-shot', 'few-shot', or 'cot'
        demo_examples (list of dicts): Demo examples with 'input' and 'target' keys
        task_name (str): Name of the task (REQUIRED for CoT technique)

    Returns:
        str: The fully formatted prompt ready for API call
    """

    if technique == 'zero-shot':
        # No examples, just the question
        return f"Q: {question}\nA:"

    elif technique == 'few-shot':
        # Few-shot WITHOUT reasoning demonstrations
        prompt_prefix = ""
        for ex in demo_examples:
            prompt_prefix += f"Q: {ex['input']}\nA: {ex['target']}\n\n"
        
        # Remove trailing newlines and add the test question
        prompt_prefix = prompt_prefix.rstrip()
        return f"{prompt_prefix}\n\nQ: {question}\nA:"

    elif technique == 'cot':
        # Load REAL CoT demonstrations from BBH cot-prompts directory
        if task_name is None:
            raise ValueError("task_name is required for CoT prompting")
        
        cot_demonstrations = load_cot_demonstrations(task_name)
        
        if cot_demonstrations is None:
            # Fallback to broken implementation if file not found
            print(f"ERROR: Could not load CoT prompts for {task_name}, using broken fallback")
            prompt_prefix = ""
            for ex in demo_examples:
                prompt_prefix += format_example(ex['input'], ex['target'], include_reasoning=True)
                prompt_prefix += "\n\n"
            return f"{prompt_prefix}Q: {question}\nA: Let's think step by step."
        
        # Use the real CoT demonstrations from BBH
        # The demonstrations already have the right format with full reasoning
        return f"{cot_demonstrations}\n\nQ: {question}\nA: Let's think step by step."

    else:
        raise ValueError(f"Unknown technique: {technique}")


# ============================================================
# Self-Test
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" PROMPT BUILDER SELF-TEST".center(70))
    print("=" * 70)

    mock_question = "What is 5+7?"
    mock_demos = [
        {'input': 'What is 2+2?', 'target': '4'},
        {'input': 'What is 3+5?', 'target': '8'},
        {'input': 'What is 6+1?', 'target': '7'},
    ]

    print("\n" + "="*70)
    print("1. ZERO-SHOT PROMPT")
    print("="*70)
    print(build_prompt(mock_question, 'zero-shot', []))

    print("\n" + "="*70)
    print("2. FEW-SHOT PROMPT (Answer-Only)")
    print("="*70)
    print(build_prompt(mock_question, 'few-shot', mock_demos))

    print("\n" + "="*70)
    print("3. TESTING REAL CoT WITH BBH PROMPTS")
    print("="*70)
    # Test with a real task that should have CoT prompts
    test_task = 'boolean_expressions'
    print(f"Testing with task: {test_task}")
    
    # Check if CoT file exists
    cot_file = f"data/bbh/cot-prompts/{test_task}.txt"
    if os.path.exists(cot_file):
        # Show first 500 chars of the real CoT prompt
        cot_prompt = build_prompt("not ( False ) is", 'cot', mock_demos, task_name=test_task)
        print("First 500 characters of CoT prompt:")
        print(cot_prompt[:500])
        print("\n... [truncated]")
        print("\n✅ CoT is now using REAL reasoning demonstrations from BBH!")
    else:
        print(f"❌ CoT file not found at {cot_file}")
        print("Make sure you're running from the project root directory")
    
    print("\n" + "="*70)