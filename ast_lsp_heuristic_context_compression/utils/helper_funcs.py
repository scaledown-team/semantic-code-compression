import json
import re
import requests

def evaluate_predictions(predictions, actual_mapping):
    """
    Evaluates the model's predictions against the actual values.

    Args:
        predictions: A dictionary containing the model's predicted values.
        actual_mapping: A dictionary with the actual mapping of masked values to their replacements.

    Returns:
        A dictionary with the evaluation results.
    """
    evaluation_results = {}
    if predictions:
        for mask_id, actual_value in actual_mapping.items():
            predicted_value = predictions.get(mask_id)
            evaluation_results[mask_id] = {
                'predicted': predicted_value,
                'actual': actual_value,
                'correct': predicted_value == actual_value
            }
    return evaluation_results

def extract_json_from_response(response_text):
    """
    Extracts the JSON block from the scaledown API response text,
    specifically looking for a JSON block within triple backticks (```json).

    Args:
        response_text: The text response from the scaledown API, potentially containing a JSON block within triple backticks and other text.

    Returns:
        A dictionary containing the parsed JSON, or None if no valid JSON block is found or parsing fails.
    """
    try:
        # Use regex to find the JSON block within triple backticks (```json)
        match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)

        if not match:
            print("Error: Could not find a JSON block within triple backticks in the response.")
            return None

        json_string = match.group(1)

        # Attempt to parse the extracted JSON string
        predictions = json.loads(json_string)
        return predictions

    except json.JSONDecodeError:
        print("Error: Could not parse JSON from the extracted string.")
        return None
    except Exception as e:
        print(f"An error occurred during JSON extraction or parsing: {e}")
        return None

def call_scaledown(prompt, context, headers, string, model="gemini-2.5-flash") -> requests.Response:
    """
    Calls the scaledown API with a vanilla configuration (scaledown.rate = 0).

    Args:
        prompt: The prompt for the API call.
        context: The context for the API call.
        model: The model to use for the API call.

    Returns:
        The response from the scaledown API.
    """
    url = "https://api.scaledown.xyz/compress/"
    payload = json.dumps({
      "context": context,
      "prompt": prompt,
      "code": string,
      "model": model,
      "scaledown": {
        "rate": 0
      }
    })

    with open("payload.json", "w") as f:
        f.write(payload)

    response = requests.request("POST", url, headers=headers, data=payload)
    return response