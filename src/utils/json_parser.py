"""Shared LLM JSON response parser."""

import re
import ast
import json


def parse_llm_json(response_text: str) -> dict | None:
    """
    Extract and parse JSON content from LLM response text.
    
    LLM responses often contain JSON wrapped in markdown code blocks or
    have formatting issues. This function attempts multiple parsing strategies
    to extract valid JSON data from various response formats.
    
    Args:
        response_text (str): Raw text response from the LLM
        
    Returns:
        dict: Parsed JSON data as a Python dictionary, or error dict if parsing fails
    """
    # Try to extract JSON from fenced code blocks.
    # Matches ```json ... ``` first, then plain ``` ... ``` as a fallback.
    json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    match = re.search(json_pattern, response_text)
    
    if match:
        json_str = match.group(1).strip()
        
        # Attempt direct JSON parsing
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Try to normalize common JSON formatting issues
            try:
                normalized_json = json_str
                # Fix single quotes around keys
                normalized_json = re.sub(r"'(\w+)':", r'"\1":', normalized_json)
                # Fix single quotes around values
                normalized_json = re.sub(r":\s*'([^']*?)'", r': "\1"', normalized_json)
                # Fix line breaks in strings
                normalized_json = re.sub(r'"\s*\n\s*"', r'""', normalized_json)
                # Fix trailing commas in objects
                normalized_json = re.sub(r',\s*\n\s*}', r'}', normalized_json)
                # Fix trailing commas in arrays
                normalized_json = re.sub(r',\s*\n\s*]', r']', normalized_json)
                
                return json.loads(normalized_json)
            except json.JSONDecodeError:
                # Try parsing as Python literal (for dict-like structures)
                try:
                    python_dict = ast.literal_eval(json_str)
                    return python_dict
                except (SyntaxError, ValueError):
                    # Final attempt: clean up whitespace and formatting
                    try:
                        cleaned_json = re.sub(r'\n\s*', ' ', json_str)
                        cleaned_json = re.sub(r',\s*}', '}', cleaned_json)
                        cleaned_json = re.sub(r',\s*]', ']', cleaned_json)
                        return json.loads(cleaned_json)
                    except json.JSONDecodeError:
                        # Return error if all parsing attempts fail
                        return {
                            "error": f"Failed to parse response as JSON: {json_str[:200]}..."
                        }
    
    # If no markdown blocks found, try parsing the entire response
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass
    
    # Return error if no valid JSON found
    return {"error": "No valid JSON content found in response"}