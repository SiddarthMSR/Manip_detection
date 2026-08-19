import os
import time
import pandas as pd
from google import genai
from google.genai import types

# API Key Setup
api_key = "AIzaSyBm005IxbipfAIe1AnMBxXEOkjzLZA6TbI" # Replace with your Google AI Studio API key
client = genai.Client(api_key=api_key)

# Model Selection
gemini_model = "gemini-2.0-flash-lite"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')

# Import Dataset
test = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))

# Constructor: Person2 Intent
def intent_p2(data):
    system_prompt = """
    I will provide you with a dialogue. 
    Please summarize the intent of the statement made by Person2 in one sentence.
    """
    
    def analyze_dialogue(dialogue):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=gemini_model,
                    contents=dialogue,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.1,
                        top_p=0.5,
                    )
                )
                time.sleep(4) # Respect the 15 RPM Free Tier limit
                return response.text.strip()
            
            except Exception as e:
                if "429" in str(e) or "quota" in str(e).lower():
                    print(f"Rate limit hit. Waiting 30s... (Attempt {attempt+1}/{max_retries})")
                    time.sleep(30)
                else:
                    print(f"Error: {e}")
                    return "Error extracting intent."
        return "Failed after retries."

    print(f"Extracting Person 2 intents using {gemini_model}...")
    data['Intent_p2'] = data['Dialogue'].apply(analyze_dialogue)
    data.to_csv(os.path.join(DATASET_DIR, 'intent2_gemini-2.0-flash-lite.csv'), index=False)
    return data

if __name__ == "__main__":
    print("------Person2 Intent------")
    intent2 = intent_p2(test)
    print(intent2.head())