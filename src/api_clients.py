import requests
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

# --- API-Specific Functions ---

def call_gemini(prompt, api_key, temperature):
    """
    Calls the Google Gemini API.
    """
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-pro:generateContent"
    headers = {"x-goog-api-key": api_key}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature}
    }
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status() # Raises an exception for bad responses (4xx or 5xx)
        return response.json()['candidates'][0]['content']['parts'][0]['text']
    except requests.exceptions.RequestException as e:
        print(f"Gemini API request failed: {e}")
        raise # Re-raise exception to be caught by tenacity
    except (KeyError, IndexError) as e:
        print(f"Error parsing Gemini response: {e}. Response: {response.text}")
        raise


def call_llama(prompt, model_name, api_key, temperature):
    """
    Calls the Groq Llama API.
    """
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e: # Catch Groq-specific or other errors
        print(f"Llama (Groq) API request failed: {e}")
        raise # Re-raise exception to be caught by tenacity


# --- Tenacity Retry Wrapper ---

@retry(
    stop=stop_after_attempt(3),                # Stop after 3 attempts
    wait=wait_exponential(multiplier=1, min=4, max=10) # Wait 4s, then 8s, then 10s
)
def api_call_with_retry(api_func, **kwargs):
    """
    Wraps any API call function with exponential backoff retry logic.

    Args:
        api_func: The API function to call (e.g., call_gemini, call_llama).
        **kwargs: Arguments to pass to the API function.

    Returns:
        The API response.
    """
    # print(f"Attempting call to {api_func.__name__}...") # Uncomment for debugging
    return api_func(**kwargs)