import json
import logging
from django.conf import settings
from groq import Groq
from . import assistant_tools

logger = logging.getLogger(__name__)

# List of tool schemas that Groq models understand
CO_PILOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_inventory_status",
            "description": "Get the current stock levels of all products on the shelf, including current count, missing count, total shelf capacity, stock percentage, and whether status is OK, Warning, or Critical.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_active_sessions",
            "description": "Get a list of all currently active (running) shelf monitoring sessions, including their session ID, camera name, and start time.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_session_summary",
            "description": "Get a detailed performance summary of a specific monitoring session or the most recent session if no session_id is specified. Returns total samples, average stock level, average FPS, latency, and list of triggered alerts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "integer",
                        "description": "Optional ID of the session to query. If omitted, queries the most recent session."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_recent_alerts",
            "description": "Get a list of the most recent alert events triggered in the system, detailing the product, severity level (Warning or Critical), and trigger time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of alerts to return (default 10)."
                    }
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_camera_configs",
            "description": "Get all configured camera inputs in the system, showing their RTSP URLs, frame skip rates, and confidence thresholds.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_history",
            "description": "Retrieve historical stock level data for a specific product over time. Crucial for plotting trends or drawing a line chart of a product's stock levels over time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The exact name of the product to query historical stock levels for, e.g. 'Coca-Cola' or 'Fanta'."
                    }
                },
                "required": ["product_name"]
            }
        }
    }
]

# Map tool names to the python functions in assistant_tools.py
TOOL_MAP = {
    'get_inventory_status': assistant_tools.get_inventory_status,
    'get_active_sessions': assistant_tools.get_active_sessions,
    'get_session_summary': assistant_tools.get_session_summary,
    'get_recent_alerts': assistant_tools.get_recent_alerts,
    'get_camera_configs': assistant_tools.get_camera_configs,
    'get_product_history': assistant_tools.get_product_history,
}

SYSTEM_PROMPT = """You are the **OSA AI Co-pilot**, a highly intelligent, context-aware assistant for the **On-Shelf Availability (OSA)** camera monitoring control center.
Your primary role is to assist operators in auditing shelf inventory, tracking active camera streaming sessions, diagnosing alert events, and inspecting historical trends.

### Tone & Style Guidelines:
1. **Professional, concise, and technical**: Keep responses direct and structured. Use Markdown tables, bold headers, and bullet points to organize data.
2. **Helpful & Proactive**: If inventory is critical, flag it clearly. Suggest potential actions (e.g. restocking or inspecting camera coordinates).
3. **Data-backed**: Rely strictly on the tools provided to query the system state. Never make up inventory numbers or camera settings. If a product is not in the system, say so clearly.
4. **Interactive Charting**: When a user asks to see history or graphs of a product's stock level over time, ALWAYS call the `get_product_history` tool. Explain the historical trends in your text response, and mention that a visual chart is displayed on their interface.

Keep your markdown clean. Do not include raw JSON dumps in your text response unless explicitly asked; format the tool data into human-readable markdown tables or lists instead.
"""

def query_co_pilot(prompt, conversation_history=None):
    """
    Core function to chat with the Groq-powered AI co-pilot.
    Handles multiple tool-calling iterations automatically.
    
    Args:
        prompt (str): The user's input message.
        conversation_history (list, optional): Previous messages in standard format.
        
    Returns:
        dict: A dictionary containing:
            - 'message': Markdown text response from the assistant.
            - 'chart_data': Optional dictionary for frontend Chart.js rendering.
            - 'tool_calls': List of tools that were invoked.
    """
    api_key = getattr(settings, 'GROQ_API_KEY', None)
    if not api_key:
        raise ValueError(
            "Groq API Key not found. Please set `GROQ_API_KEY` in your `.env` file or configure it in settings."
        )
        
    client = Groq(api_key=api_key)
    
    # Initialize message list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    if conversation_history:
        # Load up to the last 10 messages for memory efficiency
        messages.extend(conversation_history[-10:])
        
    # Append the new user prompt
    messages.append({"role": "user", "content": prompt})
    
    model_name = "llama-3.3-70b-versatile"
    fallback_model = "llama-3.1-70b-versatile"
    
    tool_calls_executed = []
    chart_data = None
    
    # Loop to allow multi-step tool calls (max 3 turns)
    for turn in range(3):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=CO_PILOT_TOOLS,
                tool_choice="auto",
                temperature=0.2,
                max_tokens=1024
            )
        except Exception as e:
            logger.error(f"Error querying Groq: {str(e)}")
            # Try a standard fallback model if the specdec is not available
            try:
                response = client.chat.completions.create(
                    model=fallback_model,
                    messages=messages,
                    tools=CO_PILOT_TOOLS,
                    tool_choice="auto",
                    temperature=0.2,
                    max_tokens=1024
                )
            except Exception as e2:
                logger.error(f"Error querying Groq fallback: {str(e2)}")
                raise RuntimeError(f"Failed to query Groq LLM: {str(e2)}")
                
        response_message = response.choices[0].message
        
        # Check if the model requested a tool call
        if response_message.tool_calls:
            # We must append the assistant message (with tool calls) to keep history consistent
            messages.append(response_message)
            
            for tool_call in response_message.tool_calls:
                func_name = tool_call.function.name
                raw_args = tool_call.function.arguments or '{}'
                try:
                    func_args = json.loads(raw_args) if raw_args.strip() else {}
                    if func_args is None:
                        func_args = {}
                except Exception as je:
                    logger.warning(f"Failed to parse arguments for tool {func_name}: {je}")
                    func_args = {}
                    
                tool_calls_executed.append(func_name)
                
                # Execute local DB tool
                if func_name in TOOL_MAP:
                    func_to_call = TOOL_MAP[func_name]
                    try:
                        tool_result = func_to_call(**func_args)
                        
                        # Special handling if this was a product history query
                        if func_name == 'get_product_history' and isinstance(tool_result, dict):
                            chart_data = tool_result
                            
                        # Add results to history
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": json.dumps(tool_result)
                        })
                    except Exception as exc:
                        logger.error(f"Error running assistant tool {func_name}: {str(exc)}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": func_name,
                            "content": json.dumps({"error": str(exc)})
                        })
                else:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": func_name,
                        "content": json.dumps({"error": f"Tool '{func_name}' is not registered."})
                    })
            # Continue the loop so the model can read tool outputs and formulate final response
            continue
        else:
            # No tool calls, this is the final text answer from the assistant
            return {
                "message": response_message.content,
                "chart_data": chart_data,
                "tool_calls": tool_calls_executed
            }
            
    # If it exceeded max turns without finishing, return the last generated assistant response
    return {
        "message": response_message.content or "I have processed your query but could not formulate a final response in time.",
        "chart_data": chart_data,
        "tool_calls": tool_calls_executed
    }
