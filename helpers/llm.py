# api_handler.py
import json
import logging
from typing import Dict, Any, Optional

# Assuming this library is in the same directory or installed
from ast_lsp_heuristic_context_compression import (
    call_scaledown,
    extract_json_from_response,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_llm_predictions(
    masked_code: str,
    context: str,
    prompt: str,
    headers: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """
    Sends the masked code and context to the Scaledown API and parses the response.

    Args:
        masked_code: The code with identifiers replaced by mask tokens.
        context: The combined content of other files in the repository.
        prompt: The system prompt for the LLM.
        headers: The request headers, including the API key.

    Returns:
        A dictionary of parsed predictions, or None if the API call fails or
        parsing is unsuccessful.
    """
    logging.info("Sending request to Scaledown API...")
    try:
        # The API call now uses the combined context and the masked target code
        response = call_scaledown(prompt=prompt, context=context, string=masked_code, headers=headers)
    except Exception as e:
        logging.error(f"An exception occurred during the API call: {e}")
        return None

    status_code = getattr(response, "status_code", None)

    if response and status_code == 200:
        logging.info("API call successful (Status 200).")
        try:
            # Extract the text content from the response object
            response_text = response.text
            if not response_text:
                # Fallback for different response structures
                response_data = response.json()
                response_text = response_data.get("full_response") or response_data.get("response")

            if response_text:
                print("Parsing predictions from API response...")
                

                response_text = json.loads(response_text) if isinstance(response_text, str) else response_text
                
                parsed_predictions = extract_json_from_response(response_text['full_response'] if 'full_response' in response_text else response_text)
                if parsed_predictions:
                    logging.info("Successfully parsed predictions from the response.")
                    return parsed_predictions
                else:
                    logging.warning("Could not parse JSON from the API response.")
                    return None
            else:
                logging.warning("API response text is empty.")
                return None
        except Exception as e:
            logging.error(f"Failed to parse API response: {e}")
            return None
    else:
        logging.error(f"API call failed with status code: {status_code}.")
        return None