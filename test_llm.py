import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tools import search_literature

load_dotenv()
sys.stdout.reconfigure(encoding='utf-8')

def run_manual_agent_loop():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set!")
        sys.exit(1)
        
    client = genai.Client(api_key=api_key)
    
    # 1. Initialize the conversation history with the user request
    messages = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text="Find recent studies about eye-movement biomarkers in Alzheimer's disease.")]
        )
    ]
    
    print("\n=== STEP 1: Sending user query + tool definitions ===")
    print(f"User Message: {messages[0].parts[0].text}")
    
    # Call Gemini, passing the function directly and disabling automatic execution
    response = client.models.generate_content(
        model='gemini-3.6-flash',
        contents=messages,
        config=types.GenerateContentConfig(
            tools=[search_literature],  # Pass Python function directly
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)  # Disable automatic execution
        )
    )
    
    # Check if Gemini wants to call a tool
    if response.function_calls:
        print("\n=== STEP 2: Gemini requested a Tool Call ===")
        tool_call = response.function_calls[0]
        print(f"Requested Tool Name: '{tool_call.name}'")
        print(f"Arguments: {tool_call.args}")
        
        # Add Gemini's tool call response to the message history.
        # This keeps the model's reasoning history intact.
        messages.append(response.candidates[0].content)
        
        # 2. Execute the tool locally on the backend
        print("\n=== STEP 3: Executing tool locally on backend ===")
        if tool_call.name == "search_literature":
            kwargs = {k: v for k, v in tool_call.args.items() if v is not None}
            tool_output = search_literature(**kwargs)
            print(f"Tool Output returned: {tool_output}")
            
            # 3. Create a Tool Response message to send back
            # The role must be 'tool' or 'function', and we link it using the tool_call.name
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
            
            # 4. Call Gemini again with the full history (User query + Tool call + Tool result)
            print("\n=== STEP 4: Sending tool results back to Gemini ===")
            final_response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=messages,
                config=types.GenerateContentConfig(
                    tools=[search_literature]
                )
            )
            
            print("\n=== STEP 5: Gemini's Final Synthesized Response ===")
            print(final_response.text)
            print("===================================================\n")
            
    else:
        print("Gemini did not call a tool. Direct response:")
        print(response.text)

if __name__ == "__main__":
    run_manual_agent_loop()
