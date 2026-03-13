"""
Quick smoke test for the LiteLLM proxy.

Usage:
    python test_proxy.py                          # uses defaults
    python test_proxy.py --model gemini-flash      # test a specific model alias
    python test_proxy.py --base-url http://localhost:4000  --key sk-change-me
"""

import argparse
import os

from openai import OpenAI


def test_chat(client: OpenAI, model: str) -> None:
    print(f"\n── Testing chat completions with model: {model}")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Say hello in one sentence."}],
        max_tokens=64,
    )
    print(f"   Response: {response.choices[0].message.content}")
    if response.usage:
        print(f"   Tokens — prompt: {response.usage.prompt_tokens}, completion: {response.usage.completion_tokens}")


def test_streaming(client: OpenAI, model: str) -> None:
    print(f"\n── Testing streaming with model: {model}")
    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Count from 1 to 5."}],
        max_tokens=64,
        stream=True,
    )
    print("   Stream: ", end="")
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print()


def test_models(client: OpenAI) -> None:
    print("\n── Available models on proxy:")
    models = client.models.list()
    for m in models.data:
        print(f"   • {m.id}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LiteLLM proxy")
    parser.add_argument("--base-url", default="http://localhost:4000", help="Proxy URL")
    parser.add_argument("--key", default=os.getenv("LITELLM_MASTER_KEY", "sk-change-me-to-something-secret"))
    parser.add_argument("--model", default=None, help="Model alias to test (omit to list available models)")
    args = parser.parse_args()

    client = OpenAI(base_url=args.base_url, api_key=args.key)

    test_models(client)

    if args.model:
        test_chat(client, args.model)
        test_streaming(client, args.model)
    else:
        print("\n   Pass --model <name> to run chat + streaming tests.")


if __name__ == "__main__":
    main()
