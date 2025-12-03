from typing import Any, Dict, List


def format_openai_messages_to_yagpt(messages: List[Dict[str, Any]]):
    messages_inner = []
    tools_cur = []

    for message in messages:
        if message["role"] == "support":  # assumption for a certain client
            message["role"] = "assistant"

        if message["role"] == "function":  # assumption for a certain client
            message["role"] = "tool"

        if message["role"] not in ["user", "assistant", "system", "tool"]:
            raise ValueError(f"Unexpected role {message['role']} in custom format")

        if message["role"] != "tool" and tools_cur:
            messages_inner.append({
                "role": "user",
                "tool_results": tools_cur
            })

        match message["role"]:
            case "system" | "user":
                messages_inner.append(message)
            case "assistant":
                if "tool_calls" in message:
                    tool_calls_inner = [
                        {
                            "name": tool_call["function"]["name"],
                            "parameters": tool_call["function"]["arguments"]
                        }
                        for tool_call in message["tool_calls"]
                    ]
                    messages_inner.append({"role": "assistant", "tool_calls": tool_calls_inner})
                else:
                    messages_inner.append(message)
            case "tool":
                if "tool_call_id" not in message:
                    raise ValueError("Tool call id not found in one of the tool calls")

                tools_cur.append({
                    "name": message["tool_call_id"],
                    "content": message["content"]
                })

    if tools_cur:
        messages_inner.append({
            "role": "user",
            "tool_results": tools_cur
        })

    return messages_inner


def format_openai_tools_to_yagpt(tools: List[Dict[str, Any]]):
    return [tool["function"] for tool in tools]
