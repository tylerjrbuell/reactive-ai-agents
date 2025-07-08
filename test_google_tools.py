#!/usr/bin/env python3
"""
Simple test script to test Google provider tool calling functionality.
"""

import asyncio
import json
import os
from reactive_agents.providers.llm.google import GoogleModelProvider


async def test_google_tool_calling():
    """Test Google provider tool calling with a simple function."""

    print("🔧 Testing Google Provider Tool Calling")
    print("=" * 50)

    # Check if API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable not set")
        return

    try:
        # Create Google provider with a lighter model
        provider = GoogleModelProvider(
            model="gemini-1.5-flash",  # Use a different model to avoid rate limits
            options={"temperature": 0.3, "max_output_tokens": 500},
        )

        # Configure more permissive safety settings to avoid blocks
        provider.configure_safety_settings()

        print(f"✅ Created Google provider with model: {provider.model}")

        # Define a simple tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city name, e.g. New York",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        print("\n🛠️ Testing simple tool calling...")

        # Test with a message that should trigger tool calling
        messages = [
            {
                "role": "user",
                "content": "What's the weather like in Paris? Use the get_weather tool.",
            }
        ]

        response = await provider.get_chat_completion(
            messages=messages, tools=tools, tool_choice="auto"
        )

        print(f"\nResponse content: {response.message.content}")
        print(f"Tool calls: {response.message.tool_calls}")

        if response.message.tool_calls:
            print("✅ Tool calling successful!")
            for tool_call in response.message.tool_calls:
                print(f"  - Tool: {tool_call['function']['name']}")
                print(f"  - Arguments: {tool_call['function']['arguments']}")
        else:
            print("❌ No tool calls generated")

        # Test without tools to compare
        print("\n📝 Testing without tools for comparison...")
        response_no_tools = await provider.get_chat_completion(
            messages=[{"role": "user", "content": "What's the weather like in Paris?"}]
        )
        print(f"Response without tools: {response_no_tools.message.content}")

    except Exception as e:
        print(f"❌ Error testing Google provider: {e}")
        import traceback

        traceback.print_exc()


async def test_google_tool_conversion():
    """Test the tool conversion logic specifically."""
    print("\n🔄 Testing Google Tool Conversion Logic")
    print("=" * 50)

    # Check if API key is available
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable not set")
        return

    try:
        provider = GoogleModelProvider(model="gemini-1.5-flash")

        # Test tool conversion
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "simple_tool",
                    "description": "A simple test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "message": {"type": "string", "description": "Test message"}
                        },
                        "required": ["message"],
                    },
                },
            }
        ]

        print("Original tools:")
        print(json.dumps(tools, indent=2))

        # Test the tool conversion logic (manually extract from get_chat_completion)
        import google.generativeai as genai  # type: ignore

        available_functions = []
        for tool in tools:
            if tool.get("type") == "function":
                func_def = tool.get("function", {})
                google_func = genai.types.FunctionDeclaration(  # type: ignore
                    name=func_def.get("name", ""),
                    description=func_def.get("description", ""),
                    parameters=func_def.get("parameters", {}),
                )
                available_functions.append(google_func)

        if available_functions:
            google_tool = genai.types.Tool(function_declarations=available_functions)  # type: ignore
            print("\n✅ Tool conversion successful!")
            print(f"Google tool object created: {type(google_tool)}")
            print(f"Function declarations: {len(google_tool.function_declarations)}")
        else:
            print("❌ Tool conversion failed")

    except Exception as e:
        print(f"❌ Error testing tool conversion: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_google_tool_calling())
    asyncio.run(test_google_tool_conversion())
