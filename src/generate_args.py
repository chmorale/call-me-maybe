import re
from typing import Any
from llm_sdk import Small_LLM_Model


def get_digit_token_ids(vocab: dict[str, int]) -> dict[str, int]:
    """Map each digit character ('0'-'9') to its token id
    in this model's vocab."""
    return ({output_text: token_id for output_text, token_id in vocab.items()
             if output_text in "0123456789"})


def get_boolean_token_ids(vocab: dict[str, int]) -> dict[str, int]:
    """Map the boolean literals ('Ġtrue', 'Ġfalse') to their token id.

    Args:
        vocab: The model's vocabulary, mapping token text to token id.

    Returns:
        dict[str, int]: The two boolean tokens mapped to their id.
    """
    boolean_texts = ("Ġtrue", "Ġfalse")
    return ({output_text: token_id for output_text, token_id in vocab.items()
             if output_text in boolean_texts})


def get_terminator_token_ids(
        vocab: dict[str, int],
        char: str) -> dict[str, int]:
    """Map every vocab token whose text contains the given character to its id.

    Args:
        vocab: The model's vocabulary, mapping token text to token id.
        char: The terminator character to look for (e.g. ',' or '}').

    Returns:
        dict[str, int]: Tokens whose text contains `char`, mapped to their id.
    """
    return {token: token_id for token, token_id in vocab.items()
            if token.strip() == char}


def get_safe_string_token_ids(vocab: dict[str, int]) -> dict[str, int]:
    """Map every vocab token safe to use unescaped inside a JSON string
    value to its id.

    A token is considered safe if its text does not contain a character
    that would break JSON string validity if inserted raw: an unescaped
    double quote, a backslash, or a raw control character such as a
    newline, carriage return, or tab.

    Args:
        vocab: The model's vocabulary, mapping token text to token id.

    Returns:
        dict[str, int]: Safe-to-use string tokens mapped to their id.
    """
    forbidden_chars = ('"', "\\", "\n", "\r", "\t")
    return {token: token_id for token, token_id in vocab.items()
            if not any(char in token for char in forbidden_chars)}


def get_quote_token_ids(vocab: dict[str, int]) -> dict[str, int]:
    """Map every vocab token containing a double quote to its id.

    These act as the terminator tokens for string generation: the string
    value ends at the first token whose text contains a `"`.

    Args:
        vocab: The model's vocabulary, mapping token text to token id.

    Returns:
        dict[str, int]: Tokens containing `"` mapped to their id.
    """
    return {token: token_id for token, token_id in vocab.items()
            if '"' in token}


def generate_string(
    model: Small_LLM_Model,
    current_input: list[int],
    vocab: dict[str, int],
    parameter_name: str,
    i: int,
) -> tuple[str, list[int]]:
    """Generate a string argument value token by token until a closing quote.

    Args:
        model: The loaded language model.
        current_input: Token ids generated so far (context + prior values).
        vocab: The model's vocabulary, mapping token text to token id.
        parameter_name: Name of the parameter being generated, used to
            build the prompt line.
        i: Index of this parameter among the function's parameters, used
            to phrase the ordinal ("first", "second", ...) in the prompt.

    Returns:
        tuple[str, list[int]]: The generated string value, and the list of
            token ids generated for it (including the closing quote token)."""

    # ordinal = "first" if i == 0 else "second" if i == 1 else f"{i+1}th"
    # parameter_line = (
    #    f'(use the {ordinal} string mentioned in the question. '
    #    f'"{parameter_name}": "'
    # )
    parameter_line = (
        f'(use the value that corresponds to "{parameter_name}" '
        f'in the question, not any other word) '
        f'"{parameter_name}": "'
    )
    ids_extra = model.encode(parameter_line)[0].tolist()
    current_input = current_input + ids_extra

    safe_ids = get_safe_string_token_ids(vocab)
    quote_ids = get_quote_token_ids(vocab)
    valid_ids = set(safe_ids.values()) | set(quote_ids.values())

    generated_ids: list[int] = []
    strings_so_far = ""
    counter = 0

    while True:
        counter += 1
        if counter > 20:
            raise RuntimeError(
                f"Could not generate a valid string for parameter "
                f"{parameter_name!r} after 20 attempts"
            )
        logits = model.get_logits_from_input_ids(current_input + generated_ids)

        best_id = max(valid_ids, key=lambda x: logits[x])

        if best_id in quote_ids.values():
            token_text = model.decode([best_id])
            content_before_quote = token_text.split('"')[0]
            strings_so_far += content_before_quote
            generated_ids.append(best_id)
            break

        generated_ids.append(best_id)
        strings_so_far += model.decode([best_id])

    return str(strings_so_far), generated_ids


def generate_boolean(
    model: Small_LLM_Model,
    current_input: list[int],
    vocab: dict[str, int],
    parameter_name: str,
    i: int,
) -> tuple[bool, list[int]]:
    """Generate a boolean argument value using constrained decoding."""

    ordinal = "first" if i == 0 else "second" if i == 1 else f"{i+1}th"
    parameter_line = (
        f'(use the {ordinal} boolean mentioned in the question) '
        f'"{parameter_name}":'
    )
    ids_extra = model.encode(parameter_line)[0].tolist()
    current_input = current_input + ids_extra

    bools_ids = get_boolean_token_ids(vocab)
    logits = model.get_logits_from_input_ids(current_input)
    valid_ids = set(bools_ids.values())
    best_id = max(valid_ids, key=lambda x: logits[x])

    text = model.decode([best_id])
    value = (text.strip() == "true")
    return value, [best_id]


def generate_number(
    model: Small_LLM_Model,
    current_input: list[int],
    vocab: dict[str, int],
    parameter_name: str,
    is_last: bool,
    i: int,
) -> tuple[float, list[int]]:
    """Generate a numeric argument value using constrained decoding.

    Args:
        model: The loaded language model.
        current_input: Token ids generated so far (context + prior values).
        vocab: The model's vocabulary, mapping token text to token id.
        is_last: Whether this is the last parameter (terminator is '}'
            instead of ',').

    Returns:
        tuple[float, list[int]]: The parsed numeric value, and the list of
            token ids generated for it (terminator token excluded).
    """
    ordinal = "first" if i == 0 else "second" if i == 1 else f"{i+1}th"
    parameter_line = (
        f'(use the {ordinal} number mentioned in the question) '
        f'"{parameter_name}":'
    )
    ids_extra = model.encode(parameter_line)[0].tolist()
    current_input = current_input + ids_extra
    digit_ids = get_digit_token_ids(vocab)
    terminator_char = "}" if is_last else ","
    terminator_ids = get_terminator_token_ids(vocab, terminator_char)

    generated_ids: list[int] = []
    digits_so_far = ""
    has_digit = False
    counter = 0

    while True:
        counter += 1
        if counter > 20:
            raise RuntimeError(
                f"Could not generate a valid number for parameter "
                f"{parameter_name!r} after 20 attempts"
                )
        logits = model.get_logits_from_input_ids(current_input + generated_ids)
        if not has_digit:
            valid_ids = set(digit_ids.values())
        else:
            valid_ids = set(digit_ids.values()) | set(terminator_ids.values())

        best_id = max(valid_ids, key=lambda x: logits[x])

        if best_id in terminator_ids.values():
            break

        generated_ids.append(best_id)
        digits_so_far += model.decode([best_id])
        has_digit = True

    return float(digits_so_far), generated_ids


def generate_args(
    model: Small_LLM_Model,
    best_function: dict[str, Any],
    prompt_item: str,
    vocab: dict[str, int],
) -> dict[str, float | bool | str | int]:
    """Generate the args dict for the chosen function,
       one parameter at a time."""
    numbers = re.findall(r'-?\d+\.?\d*', prompt_item)
    basic_context = (
        f'User question: "{prompt_item}"\n'
        f'Function: {best_function["name"]} — {best_function["description"]}\n'
        f'Extract the argument values directly from the user question above.\n'
        f'in the order they appear. Each parameter must use a different value '
        f'from the question.\n'
        f'Numbers mentioned in the question, in order: {", ".join(numbers)}.\n'
        f'Generate a JSON object with the argument values.\n'
        f'{{'
    )
    current_input = model.encode(basic_context)[0].tolist()

    parameters = best_function["parameters"]
    parameter_names = list(parameters.keys())
    args: dict[str, float | bool | str | int] = {}

    for i, name in enumerate(parameter_names):
        param_type = parameters[name]["type"]
        is_last = (i == len(parameter_names) - 1)
        value: float | bool | str | int

        if param_type == "number":
            value, generated_tokens = generate_number(
                model, current_input, vocab, name, is_last, i
            )
        elif param_type == "integer":
            value, generated_tokens = generate_number(
                model, current_input, vocab, name, is_last, i
            )
            value = int(value)
        elif param_type == "boolean":
            value, generated_tokens = generate_boolean(
                model, current_input, vocab, name, i
            )
        elif param_type == "string":
            value, generated_tokens = generate_string(
                model, current_input, vocab, name, i
            )
        else:
            raise ValueError(f"Unsupported parameter type: {param_type!r}")

        args[name] = value
        current_input = current_input + generated_tokens

    return args
