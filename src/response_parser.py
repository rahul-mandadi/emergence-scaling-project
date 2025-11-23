import re

def extract_answer(response_text, task_name):
    """
    Production-ready answer extractor with support for:
    - Boolean expressions (True/False with markdown)
    - Multiple-choice (A-K with various formats)
    - Word sorting (comma-separated, numbered lists, or prose)
    
    Args:
        response_text (str): The raw model output
        task_name (str): The BBH task identifier
        
    Returns:
        str: The extracted answer, or empty string if parsing fails
    """
    # Handle NaN, None, or empty responses
    if response_text is None or response_text == '':
        return ""
    
    # Handle pandas NaN (which is a float)
    if isinstance(response_text, float):
        import math
        if math.isnan(response_text):
            return ""
    
    # Convert to string if not already (safety check)
    response_text = str(response_text).strip()
    
    if not response_text:
        return ""
    

    
    # ============================================================
    # PRIORITY 1: Boolean Expressions (Check FIRST)
    # ============================================================
    if task_name == 'boolean_expressions':
        patterns = [
            r'(?:answer|result|final answer|final result|conclusion|therefore)[,\s]*(?:is|:)?\s*\*?\*?\s*\b(True|False)\b',
            r'(?:evaluates to|becomes|equals)[,\s]*\*?\*?\s*\b(True|False)\b',
            r'(?:so|thus|hence)[,\s]+(?:the answer is|it is|the result is)[,\s]*\*?\*?\s*\b(True|False)\b',
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                return match.group(1).capitalize()
        
        last_part = response_text[-150:]
        true_matches = list(re.finditer(r'\bTrue\b', last_part, re.IGNORECASE))
        false_matches = list(re.finditer(r'\bFalse\b', last_part, re.IGNORECASE))
        
        last_true_pos = true_matches[-1].start() if true_matches else -1
        last_false_pos = false_matches[-1].start() if false_matches else -1
        
        if last_true_pos > last_false_pos:
            return "True"
        elif last_false_pos > last_true_pos:
            return "False"
    
    # ============================================================
    # PRIORITY 2: Multiple-Choice (A, B, C...)
    # ============================================================
    if task_name in ['date_understanding', 
                     'tracking_shuffled_objects_five_objects', 
                     'geometric_shapes']:
        
        patterns = [
            r'(?:answer|option|correct answer|correct option|conclusion) is[:\s]*\(?([A-K])\)?',
            r'(?:so|therefore|thus|hence)[,\s]+(?:the answer is|it is)[,\s]*\(?([A-K])\)?',
            r'\(([A-K])\)',
            r'\*\*\(([A-K])\)\*\*',
            r'final answer[:\s]*(?:is[:\s]*)?\$?\\?boxed\{?\(?([A-K])\)?\}?\$?',
            r'^\s*\*?\*?\s*\(?([A-K])\)?\s*\*?\*?\s*$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.MULTILINE | re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        cleaned = response_text.strip('*()').strip()
        if len(cleaned) == 1 and cleaned.upper() in 'ABCDEFGHIJK':
            return cleaned.upper()

    # ============================================================
    # PRIORITY 3: Word Sorting (FULLY FIXED)
    # ============================================================
    # ============================================================
    # PRIORITY 3: Word Sorting (FULLY FIXED)
    # ============================================================
    if task_name == 'word_sorting':
        lines = response_text.split('\n')
        
        # Pattern 1: Numbered list (e.g., "1. bedtime\n2. boon")
        numbered_items = []
        for line in lines:
            match = re.match(r'^\s*\d+[\.\)]\s+([a-zA-Z]\w+)', line.strip())
            if match:
                numbered_items.append(match.group(1))
        
        if len(numbered_items) >= 4:
            return ' '.join(numbered_items)
        
        # Pattern 2: Line-by-line words (e.g., "apple\nbanana\ncherry")
        single_word_lines = []
        for line in lines:
            stripped = line.strip()
            # Is this line a single alphabetic word (2+ chars)?
            if stripped and stripped.isalpha() and len(stripped) > 1:
                single_word_lines.append(stripped)
        
        if len(single_word_lines) >= 4:
            return ' '.join(single_word_lines)
        
        # Pattern 3: Extract word list from anywhere in text
        # Look for sequences of 4+ words separated by spaces/commas
        # This handles "Here are the words in order: apple banana cherry date"
        for line in reversed(lines[-15:]):
            if len(line.strip()) < 20:
                continue
            
            # Extract ALL alphabetic words (2+ chars) from the line
            all_words = re.findall(r'\b[a-zA-Z]{2,}\b', line)
            
            if len(all_words) < 4:
                continue
            
            # Filter out common English stop words and task-related words
            stopwords = {
                'the', 'is', 'are', 'in', 'order', 'list', 'sorted', 'words', 
                'here', 'following', 'final', 'answer', 'alphabetically', 
                'alphabetical', 'these', 'correct', 'now', 'below'
            }
            
            filtered_words = [w for w in all_words if w.lower() not in stopwords]
            
            # If we have 4+ content words after filtering, return them
            if len(filtered_words) >= 4:
                return ' '.join(filtered_words)
        
        # Pattern 4: Comma-separated list with prefix
        match = re.search(r'(?:answer|list|result):?\s*([\w\s,]+)', response_text, re.IGNORECASE)
        if match:
            extracted = match.group(1).strip().strip(',')
            words = extracted.replace(',', ' ').split()
            alphabetic_words = [w for w in words if w.isalpha() and len(w) > 1]
            if len(alphabetic_words) >= 4:
                return ' '.join(alphabetic_words)
    
    # ============================================================
    # General Fallback
    # ============================================================
    match = re.search(r'answer:\s*(.*)', response_text, re.IGNORECASE | re.DOTALL)
    if match:
        try:
            answer_part = match.group(1).strip()
            first_line = [line.strip() for line in answer_part.split('\n') if line.strip()][0]
            return first_line.strip().strip(',')
        except (IndexError, AttributeError):
            pass
    
    return ""


# ============================================================
# Self-Test
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" RESPONSE PARSER SELF-TEST".center(70))
    print("=" * 70)
    
    test_cases = [
        # Boolean expressions
        {
            'task': 'boolean_expressions',
            'response': 'The final result is **True**.',
            'expected': 'True',
            'label': 'Boolean with markdown'
        },
        {
            'task': 'boolean_expressions',
            'response': 'Step 1: not True = False\nStep 2: False or True = True\nTherefore, the answer is False',
            'expected': 'False',
            'label': 'Boolean CoT format'
        },
        {
            'task': 'boolean_expressions',
            'response': 'So the result is True.',
            'expected': 'True',
            'label': 'Boolean simple'
        },
        
        # Multiple choice
        {
            'task': 'date_understanding',
            'response': 'So the answer is (C).',
            'expected': 'C',
            'label': 'MC standard'
        },
        {
            'task': 'geometric_shapes',
            'response': 'Therefore, the correct option is (F).',
            'expected': 'F',
            'label': 'MC verbose'
        },
        {
            'task': 'date_understanding',
            'response': 'The final answer is: $\\boxed{(D)}$',
            'expected': 'D',
            'label': 'MC LaTeX'
        },
        
        # Word sorting (ALL FORMATS)
        {
            'task': 'word_sorting',
            'response': 'Answer: abdominal, address, berry, bounty',
            'expected': 'abdominal address berry bounty',
            'label': 'Word: comma-separated'
        },
        {
            'task': 'word_sorting',
            'response': 'The sorted list is:\n1. bedtime\n2. boon\n3. bottle\n4. chapati\n5. kenney',
            'expected': 'bedtime boon bottle chapati kenney',
            'label': 'Word: numbered list'
        },
        {
            'task': 'word_sorting',
            'response': 'Here are the words in order: apple banana cherry date',
            'expected': 'apple banana cherry date',
            'label': 'Word: prose format'
        },
        {
            'task': 'word_sorting',
            'response': 'Sorted alphabetically:\napple\nbanana\ncherry\ndate\nelder',
            'expected': 'apple banana cherry date elder',
            'label': 'Word: line-by-line'
        },
    ]
    
    passed = failed = 0
    
    for i, test in enumerate(test_cases, 1):
        result = extract_answer(test['response'], test['task'])
        
        if test['task'] == 'word_sorting':
            result_normalized = ' '.join(result.split())
            expected_normalized = ' '.join(test['expected'].split())
            match = result_normalized == expected_normalized
        else:
            match = result == test['expected']
        
        status = "✅ PASS" if match else "❌ FAIL"
        
        if match:
            passed += 1
        else:
            failed += 1
            
        print(f"{status} - {test['label']}")
        if not match:
            print(f"       Expected: '{test['expected']}'")
            print(f"       Got:      '{result}'")
    
    print("\n" + "=" * 70)
    print(f" FINAL SCORE: {passed}/{len(test_cases)} tests passed".center(70))
    print("=" * 70)
    
    if failed == 0:
        print("\n🎉 All tests passed! Parser is ready for production.\n")
    else:
        print(f"\n⚠️  {failed} test(s) failed. Review the code.\n")