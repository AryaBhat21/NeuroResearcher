import os
from google import genai
from google.genai import types
from tools import search_literature, get_paper

# Initialize Gemini Client
# We retrieve the key from environment variables (loaded in main.py)
def get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set!")
    return genai.Client(api_key=api_key)

def run_agent_flow(user_message: str) -> str:
    """
    Runs the LLM + Tool calling lifecycle for a given user message.
    """
    client = get_client()
    
    # 1. Initialize message history
    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_message)]
        )
    ]
    
    # 2. Call model with all tools available, automatic calling disabled
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[search_literature, get_paper],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
    )
    
    # 3. Check if model wants to call a tool
    if response.function_calls:
        tool_call = response.function_calls[0]
        
        # Save model's tool call response in history
        messages.append(response.candidates[0].content)
        
        # Execute the correct tool
        tool_output = None
        if tool_call.name == "search_literature":
            kwargs = {k: v for k, v in tool_call.args.items() if v is not None}
            tool_output = search_literature(**kwargs)
        elif tool_call.name == "get_paper":
            paper_id_arg = str(tool_call.args.get("paper_id"))
            tool_output = get_paper(paper_id=paper_id_arg)
            
        if tool_output is not None:
            # Format function response and add to history
            tool_response_part = types.Part.from_function_response(
                name=tool_call.name,
                response=tool_output
            )
            messages.append(
                types.Content(
                    role="user",
                    parts=[tool_response_part]
                )
            )
            
            # 4. Request the final response from model
            final_response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=messages,
                config=types.GenerateContentConfig(
                    tools=[search_literature, get_paper]
                )
            )
            return final_response.text
    
    # If no tool was requested, return the direct text response
    return response.text
