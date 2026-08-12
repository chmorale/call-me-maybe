import json
from typing import Any
from llm_sdk import Small_LLM_Model
from .get_score import get_score
from .generate_args import generate_args


def get_context(prompt_item: str, function_list: list[dict[str, Any]]) -> str:
    """Build the context text listing available functions and the user prompt.

    Args:
        prompt_item: The natural language prompt from the user.
        function_list: List of available function definitions.

    Returns:
        str: A single text block with all functions and the prompt, ready
            to be tokenized by the model.
    """
    function_lines = ["Select the best function to solve this prompt: "]
    for func in function_list:
        line = f"- {func['name']}: {func['description']}"
        function_lines.append(line)

    function_block = "\n".join(function_lines)

    context = (
        f"Available functions:\n{function_block}\n\n"
        f"User question: \"{prompt_item}\"\n"
        f"Function to call:"
    )

    return context


def select_function(
        prompt_list: list[str],
        function_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    try:
        model = Small_LLM_Model()
    except Exception as e:
        raise RuntimeError(
            f"Could not load the LLM model: {e}"
        ) from e

    try:
        with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as f:
            vocab = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"Could not load the model vocabulary file: {e}"
        ) from e

    results = []

    for prompt_item in prompt_list:
        best_function: dict[str, Any] | None = None
        best_score = float('-inf')
        context = get_context(prompt_item, function_list)
        for function_item in function_list:
            try:
                score = get_score(model, context, function_item)
            except KeyError as e:
                print(
                    f"Warning: could not score function "
                    f"{function_item.get('name', '?')!r} for prompt "
                    f"{prompt_item!r}: missing key {e}"
                )
                continue
            if score > best_score:
                best_score = score
                best_function = function_item
        if best_function is None:
            print(f"No matching function found for prompt {prompt_item!r}")
            results.append({
                "prompt": prompt_item,
                "name": "",
                "parameters": {},
            })
            continue

        try:
            args = generate_args(model, best_function, prompt_item, vocab)
        except (RuntimeError, ValueError) as e:
            print(f"Error generating args for prompt {prompt_item!r}: {e}")
            results.append({
                "prompt": prompt_item,
                "name": best_function["name"],
                "parameters": {},
            })
            continue

        print(f"Processed: {prompt_item!r} -> {best_function['name']}")
        results.append({
            "prompt": prompt_item,
            "name": best_function['name'],
            "parameters": args,
        })

    return results
