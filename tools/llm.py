import os
import time
import logging

import litellm
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = (
    "gemini/gemini-3.8-flash,"
    "gemini/gemini-3.5-flash,"
    "groq/openai/gpt-oss-120b"
)
FALLBACK_MODELS = [
    m.strip() for m in os.getenv("LLM_MODELS", DEFAULT_MODELS).split(",") if m.strip()
]
LLM_TIMEOUT = 60
LLM_RETRIES = 3

PERMANENT_ERRORS = (
    litellm.NotFoundError,
    litellm.AuthenticationError,
    litellm.BadRequestError,
)


def call_llm(
    messages: list,
    retries: int = LLM_RETRIES,
    timeout: int = LLM_TIMEOUT,
    json_mode: bool = False,
    max_tokens: int | None = None,
) -> str:
    kwargs = {"timeout": timeout}
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    last_error = None
    for model in FALLBACK_MODELS:
        for attempt in range(retries):
            try:
                response = litellm.completion(model=model, messages=messages, **kwargs)
                content = response.choices[0].message.content or ""
                if not content.strip():
                    raise ValueError("empty response")
                return content
            except PERMANENT_ERRORS as e:
                last_error = e
                logging.warning(f"{model}: {type(e).__name__}, skipping to next model")
                break
            except ValueError as e:
                last_error = e
                logging.warning(f"{model}: {e}, skipping to next model")
                break
            except Exception as e:
                last_error = e
                logging.warning(
                    f"{model} failed (attempt {attempt + 1}/{retries}): {type(e).__name__}"
                )
                if attempt < retries - 1:
                    time.sleep(2 ** (attempt + 1)) 
        logging.info(f"Moving on from {model}")

    raise RuntimeError(f"All models failed: {type(last_error).__name__}")


if __name__ == "__main__":
    print("models:", FALLBACK_MODELS)
    print(call_llm([{"role": "user", "content": "Reply with the single word: ok"}]))