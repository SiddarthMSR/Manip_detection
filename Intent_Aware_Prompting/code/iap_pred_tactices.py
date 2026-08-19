import os
import time
import json
import pandas as pd
from google import genai
from google.genai import types
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# API Key Setup
api_key = "AIzaSyBm005IxbipfAIe1AnMBxXEOkjzLZA6TbI" # Replace with your Google API key
client = genai.Client(api_key=api_key)

gemini_model = "gemini-2.0-flash-lite"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'Dataset')

# Import Dataset
test = pd.read_csv(os.path.join(DATASET_DIR, 'test.csv'))
intent1 = pd.read_csv(os.path.join(DATASET_DIR, 'intent1_gemini-2.0-flash-lite.csv'))
intent2 = pd.read_csv(os.path.join(DATASET_DIR, 'intent2_gemini-2.0-flash-lite.csv'))

# Prepare Dataset
test['Intent_p1'] = intent1['Intent_p1']
test['Intent_p2'] = intent2['Intent_p2']

# Define Target Techniques
TECHNIQUES_LIST = [
    "Denial", "Evasion", "Feigning Innocence", "Rationalization", 
    "Playing the Victim Role", "Playing the Servant Role", 
    "Shaming or Belittlement", "Intimidation", "Brandishing Anger", 
    "Accusation", "Persuasion or Seduction"
]

def iap_prompting_with_tactics(dialogue, intent_p1, intent_p2):
    system_prompt = f"""
    I will provide you with a dialogue, the intent of person1, and the intent of person2. 
    Please carefully analyze the dialogue and intents, and determine if it contains elements of mental manipulation.
    
    If manipulation is present, classify the manipulation into one or more of these techniques:
    {', '.join(TECHNIQUES_LIST)}

    Return ONLY a JSON object with two keys:
    1. "is_manipulative": Answer 'Yes' or 'No'.
    2. "techniques": A list of strings containing the techniques identified (leave empty [] if No).
    """
    
    user_input = f"Dialogue: {dialogue}\nIntent P1: {intent_p1}\nIntent P2: {intent_p2}"
    
    # We use Google GenAI natively enforcing structured JSON output
    response = client.models.generate_content(
        model=gemini_model,
        contents=user_input,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.1,
            top_p=0.5,
            response_mime_type="application/json" 
        )
    )
    
    res_json = json.loads(response.text.strip())
    is_manipulative = res_json.get("is_manipulative", "No")
    techniques = res_json.get("techniques", [])
    
    binary_label = 1 if 'yes' in str(is_manipulative).lower() else 0
    techniques_str = ", ".join(techniques) if techniques else "None"
    
    return binary_label, techniques_str

def iap_prediction(test_data):
    targets = [int(v) for v in test_data['Manipulative'].values]
    preds = []
    
    for idx, row in test_data.iterrows():
        intent_p1 = row['Intent_p1']
        intent_p2 = row['Intent_p2']
        dialogue = row['Dialogue']
        
        try:
            pred_binary, pred_tactics = iap_prompting_with_tactics(dialogue, intent_p1, intent_p2)
            time.sleep(4) # Rate limit protection
        except Exception as e:
            if "429" in str(e):
                print(f"Rate limit hit at row {idx}. Sleeping for 30 seconds...")
                time.sleep(30)
                try: # One retry
                    pred_binary, pred_tactics = iap_prompting_with_tactics(dialogue, intent_p1, intent_p2)
                except:
                    pred_binary, pred_tactics = 0, "Error"
            else:
                print(f"Error processing row {idx}: {e}")
                pred_binary, pred_tactics = 0, "Error"
            
        preds.append(pred_binary)
        
        test_data.at[idx, 'Prediction'] = pred_binary
        test_data.at[idx, 'Predicted_Techniques'] = pred_tactics
        
        print(f"Row {idx} -> Binary: {pred_binary} | Tactics: {pred_tactics}")

    test_data.to_csv(os.path.join(DATASET_DIR, 'iap_prediction_with_tactics_gemini.csv'), index=False)
    
    # Performance Indicators
    accuracy = accuracy_score(targets, preds)
    precision = precision_score(targets, preds, zero_division=0)
    recall = recall_score(targets, preds, zero_division=0)
    weighted_f1 = f1_score(targets, preds, average='weighted', zero_division=0)
    macro_f1 = f1_score(targets, preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(targets, preds)
    
    print(f"\n- Accuracy = {accuracy:.3f}")
    print(f"- Precision = {precision:.3f}")
    print(f"- Recall = {recall:.3f}")
    print(f"- Weighted F1-Score = {weighted_f1:.3f}")
    print(f"- Macro F1-Score = {macro_f1:.3f}")
    print(f"- Confusion Matrix = \n{conf_matrix}")
    print("\nFile saved successfully to 'iap_prediction_with_tactics_gemini.csv'")

if __name__ == "__main__":
    print(f"------Experiment: IAP with Tactic Sub-classification ({gemini_model})------")
    iap_prediction(test)