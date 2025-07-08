#!/usr/bin/env python3
"""
Demo script showing how to use OpenAI and Anthropic providers.

This script demonstrates the new OpenAI and Anthropic model providers
and how to use them with the reactive agents framework.

Before running this script, make sure to:
1. Install dependencies: pip install openai anthropic
2. Set environment variables:
   - OPENAI_API_KEY=your_openai_api_key
   - ANTHROPIC_API_KEY=your_anthropic_api_key
"""

import asyncio
import os
from reactive_agents.providers.llm.factory import ModelProviderFactory
from reactive_agents.providers.llm.openai import OpenAIModelProvider
from reactive_agents.providers.llm.anthropic import AnthropicModelProvider
from reactive_agents.providers.llm.google import GoogleModelProvider


async def demo_openai_provider():
    """Demo using OpenAI provider directly."""
    print("🤖 OpenAI Provider Demo")
    print("=" * 50)

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set")
        return

    try:
        # Create OpenAI provider directly
        provider = OpenAIModelProvider(
            model="gpt-4", options={"temperature": 0.7, "max_tokens": 100}
        )
        print(f"✅ Created OpenAI provider with model: {provider.model}")

        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ]

        print("\n📝 Testing chat completion...")
        response = await provider.get_chat_completion(messages=messages)
        print(f"Response: {response.message.content}")
        print(f"Model: {response.model}")
        print(
            f"Tokens - Prompt: {response.prompt_tokens}, Completion: {response.completion_tokens}"
        )

        # Test text completion
        print("\n📝 Testing text completion...")
        response = await provider.get_completion(
            prompt="The best programming language is",
            system="You are a helpful assistant.",
        )
        print(f"Response: {response.message.content}")

    except Exception as e:
        print(f"❌ OpenAI Provider Error: {e}")


async def demo_anthropic_provider():
    """Demo using Anthropic provider directly."""
    print("\n🤖 Anthropic Provider Demo")
    print("=" * 50)

    # Check if API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("❌ ANTHROPIC_API_KEY environment variable not set")
        return

    try:
        # Create Anthropic provider directly
        provider = AnthropicModelProvider(
            model="claude-3-sonnet-20240229",
            options={"temperature": 0.7, "max_tokens": 100},
        )
        print(f"✅ Created Anthropic provider with model: {provider.model}")
        print(f"Claude 3 model: {provider.is_claude_3}")

        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Japan?"},
        ]

        print("\n📝 Testing chat completion...")
        response = await provider.get_chat_completion(messages=messages)
        print(f"Response: {response.message.content}")
        print(f"Model: {response.model}")
        print(
            f"Tokens - Prompt: {response.prompt_tokens}, Completion: {response.completion_tokens}"
        )

        # Test text completion
        print("\n📝 Testing text completion...")
        response = await provider.get_completion(
            prompt="The most interesting thing about AI is",
            system="You are a helpful assistant.",
        )
        print(f"Response: {response.message.content}")

    except Exception as e:
        print(f"❌ Anthropic Provider Error: {e}")


async def demo_google_provider():
    """Demo using Google provider directly."""
    print("\n🤖 Google Provider Demo")
    print("=" * 50)

    # Check if API key is set
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable not set")
        return

    try:
        # Create Google provider directly
        provider = GoogleModelProvider(
            model="gemini-pro", options={"temperature": 0.7, "max_output_tokens": 100}
        )
        print(f"✅ Created Google provider with model: {provider.model}")

        # Test chat completion
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Italy?"},
        ]

        print("\n📝 Testing chat completion...")
        response = await provider.get_chat_completion(messages=messages)
        print(f"Response: {response.message.content}")
        print(f"Model: {response.model}")
        print(
            f"Tokens - Prompt: {response.prompt_tokens}, Completion: {response.completion_tokens}"
        )

        # Test text completion
        print("\n📝 Testing text completion...")
        response = await provider.get_completion(
            prompt="The most fascinating aspect of space exploration is",
            system="You are a helpful assistant.",
        )
        print(f"Response: {response.message.content}")

        # Demo safety filter handling
        print("\n🛡️ Testing safety filter handling...")
        try:
            # This might trigger safety filters
            response = await provider.get_chat_completion(
                messages=[
                    {"role": "user", "content": "Tell me about controversial topics"}
                ]
            )

            if response.done_reason == "content_filter":
                print(
                    "Response was blocked by safety filters. Trying with more permissive settings..."
                )

                # Configure more permissive safety settings
                provider.configure_safety_settings()  # This disables safety filtering

                # Retry the request
                response = await provider.get_chat_completion(
                    messages=[
                        {
                            "role": "user",
                            "content": "Tell me about controversial topics",
                        }
                    ]
                )

                print(
                    f"Response after adjusting safety settings: {response.message.content}"
                )
            else:
                print(f"Response: {response.message.content}")

        except Exception as safety_error:
            print(f"Safety handling error: {safety_error}")

    except Exception as e:
        print(f"❌ Google Provider Error: {e}")


async def demo_provider_factory():
    """Demo using providers through the factory."""
    print("\n🏭 Provider Factory Demo")
    print("=" * 50)

    # List of providers to test
    providers_to_test = [
        ("openai:gpt-4", "OPENAI_API_KEY"),
        ("openai:gpt-3.5-turbo", "OPENAI_API_KEY"),
        ("anthropic:claude-3-sonnet-20240229", "ANTHROPIC_API_KEY"),
        ("anthropic:claude-3-haiku-20240307", "ANTHROPIC_API_KEY"),
        ("google:gemini-pro", "GOOGLE_API_KEY"),
        ("google:gemini-1.5-pro", "GOOGLE_API_KEY"),
    ]

    for provider_name, api_key_name in providers_to_test:
        if not os.getenv(api_key_name):
            print(f"⏭️  Skipping {provider_name} (no {api_key_name} set)")
            continue

        try:
            print(f"\n🔧 Testing {provider_name}...")
            provider = ModelProviderFactory.get_model_provider(provider_name)
            print(f"✅ Created provider: {provider.__class__.__name__}")
            print(f"   Model: {provider.model}")
            print(f"   Provider Name: {provider.name}")

            # Test simple completion
            response = await provider.get_completion(
                prompt="Hello, world!", options={"max_tokens": 50}
            )
            print(f"   Response: {response.message.content[:50]}...")

        except Exception as e:
            print(f"❌ Error with {provider_name}: {e}")


async def demo_json_output_mode():
    """Demo JSON output mode with the new providers."""
    print("\n📄 JSON Output Mode Demo")
    print("=" * 50)

    # Test with OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            print("\n🤖 Testing OpenAI JSON output mode...")
            provider = OpenAIModelProvider(model="gpt-4")

            response = await provider.get_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "List the top 3 programming languages with their popularity scores. Respond in JSON format with 'languages' array containing objects with 'name' and 'score' fields.",
                    },
                ],
                format="json",
            )

            print("OpenAI JSON Response:")
            print(response.message.content)
            print(f"Model: {response.model}")

        except Exception as e:
            print(f"❌ OpenAI JSON mode error: {e}")

    # Test with Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            print("\n🤖 Testing Anthropic JSON output mode...")
            provider = AnthropicModelProvider(model="claude-3-sonnet-20240229")

            response = await provider.get_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "Generate a JSON object with sample user profile data including name, age, and email.",
                    },
                ],
                format="json",
            )

            print("Anthropic JSON Response:")
            print(response.message.content)
            print(f"Model: {response.model}")

        except Exception as e:
            print(f"❌ Anthropic JSON mode error: {e}")

    # Test with Google
    if os.getenv("GOOGLE_API_KEY"):
        try:
            print("\n🤖 Testing Google JSON output mode...")
            provider = GoogleModelProvider(model="gemini-pro")

            response = await provider.get_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": "Create a JSON object representing a simple todo item with id, title, completed status, and priority.",
                    },
                ],
                format="json",
            )

            print("Google JSON Response:")
            print(response.message.content)
            print(f"Model: {response.model}")

        except Exception as e:
            print(f"❌ Google JSON mode error: {e}")


async def demo_tool_usage():
    """Demo tool usage with the new providers."""
    print("\n🛠️ Tool Usage Demo")
    print("=" * 50)

    # Simple tool definition
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Test with OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            print("\n🤖 Testing OpenAI tool usage...")
            provider = OpenAIModelProvider(model="gpt-4")

            messages = [
                {"role": "user", "content": "What's the weather like in New York?"}
            ]

            response = await provider.get_chat_completion(
                messages=messages, tools=tools, tool_choice="auto"
            )

            print(f"Response: {response.message.content}")
            if response.message.tool_calls:
                print(f"Tool calls: {len(response.message.tool_calls)}")
                for tool_call in response.message.tool_calls:
                    print(
                        f"  - {tool_call['function']['name']}: {tool_call['function']['arguments']}"
                    )

        except Exception as e:
            print(f"❌ OpenAI tool usage error: {e}")

    # Test with Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        try:
            print("\n🤖 Testing Anthropic tool usage...")
            provider = AnthropicModelProvider(model="claude-3-sonnet-20240229")

            messages = [
                {"role": "user", "content": "What's the weather like in Los Angeles?"}
            ]

            response = await provider.get_chat_completion(
                messages=messages, tools=tools, tool_choice="auto"
            )

            print(f"Response: {response.message.content}")
            if response.message.tool_calls:
                print(f"Tool calls: {len(response.message.tool_calls)}")
                for tool_call in response.message.tool_calls:
                    print(
                        f"  - {tool_call['function']['name']}: {tool_call['function']['arguments']}"
                    )

        except Exception as e:
            print(f"❌ Anthropic tool usage error: {e}")


async def main():
    """Main demo function."""
    print("🚀 OpenAI, Anthropic, and Google Providers Demo")
    print("=" * 60)

    # Check for API keys
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_google = bool(os.getenv("GOOGLE_API_KEY"))

    print(f"OpenAI API Key: {'✅ Set' if has_openai else '❌ Not set'}")
    print(f"Anthropic API Key: {'✅ Set' if has_anthropic else '❌ Not set'}")
    print(f"Google API Key: {'✅ Set' if has_google else '❌ Not set'}")

    if not has_openai and not has_anthropic and not has_google:
        print(
            "\n⚠️  No API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, and/or GOOGLE_API_KEY"
        )
        print("   to test the providers.")
        return

    # Run demos
    await demo_openai_provider()
    await demo_anthropic_provider()
    await demo_google_provider()
    await demo_provider_factory()
    await demo_json_output_mode()
    await demo_tool_usage()

    print("\n🎉 Demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
