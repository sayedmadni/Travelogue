from transformers import pipeline
from detoxify import Detoxify
import re

# Load the open-source moderation model
moderator = pipeline(
    "text-classification",
    #model="ProtectAI/deberta-v3-base-moderation",
    model="unitary/toxic-bert",
    top_k=None
)

# ---- Rule-based filters ----
PROFANITY_LIST = ["idiot", "stupid", "hate", "kill", "dumb"]
MAX_LENGTH = 500


def check_basic_rules(text: str) -> list:
    """Simple keyword and format-based guardrails."""
    issues = []
    if len(text) > MAX_LENGTH:
        issues.append("Input too long.")
    if any(bad in text.lower() for bad in PROFANITY_LIST):
        issues.append("Contains profanity or abusive language.")
    if re.search(r"(drop\s+table|delete\s+from|;--)", text.lower()):
        issues.append("Potential SQL injection pattern detected.")
    
    return issues


def check_toxicity(text, model_type='original', threshold=0.5):
    """
    Checks for toxicity in a given text using the detoxify library and returns 
    the categories detected above a specified threshold.

    Args:
        text (str): The text to check for toxicity.
        model_type (str): The name of the pre-trained detoxify model to use.
                           Options: 'original', 'unbiased', 'multilingual'.
                           Defaults to 'original'.
        threshold (float): The minimum confidence score to consider a category as detected.
                           Defaults to 0.5.

    Returns:
        dict: A dictionary of detected toxicity categories and their scores.
              Returns an empty dictionary if no toxicity is detected above the threshold.
    """
    try:
        model = Detoxify(model_type)
        results = model.predict(text)
        
        detected_toxicity = {
            category: score
            for category, score in results.items()
            if score > threshold
        }
        
        return detected_toxicity
    
    except ValueError as e:
        print(f"Error: {e}")
        return {}


def validate_input(text: str):
    """Combine rule-based and AI-based guardrails."""
    issues = check_basic_rules(text)
    check_toxicity_result = check_toxicity(text, threshold=0.5)

    if check_toxicity_result:
        print("Detected Toxicity Categories:")
    for category, score in check_toxicity_result.items():
        print(f"  - {category}: {score:.4f}")
        return False
    else:
        print("No significant toxicity detected.")
        

    if issues:
        print("Input blocked due to:")
        for i in issues:
            print(f" - {i}")
        return False
    else:
        print("Input passed moderation.")
        return True


# ---- Example usage ----
if __name__ == "__main__":
    user_input = input("Enter your query: ")
    validate_input(user_input)
