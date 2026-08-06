import re
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


def generate_string(
    model: Small_LLM_Model,
    current_input: list[int],
    vocab: dict[str, int],
    parameter_name: str,
    i: int,
) -> tuple[str, list[int]]:
    """"""

    ordinal = "first" if i == 0 else "second" if i == 1 else f"{i+1}th"
    parameter_line = (
        f'(use the {ordinal} string mentioned in the question. '
        f'"{parameter_name}": "'
    )
    ids_extra = model.encode(parameter_line)[0].tolist()
    current_input = current_input + ids_extra

    generated_ids: list[int] = []
    strings_so_far = ""
    intentos = 0

    while True:
        intentos += 1
        if intentos > 20:
            print("    [debug] ¡Demasiados intentos, abortando bucle!")
            break
        logits = model.get_logits_from_input_ids(current_input + generated_ids)

        best_id = max(range(len(logits)), key=lambda x: logits[x])

        print(f"    [debug] best_id={best_id} "
              f"texto={model.decode([best_id])!r} "
              )
        # print(f"    [debug] parameter_line = {parameter_line!r}")
        if '"' in model.decode([best_id]):
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

    print(f"    [debug] best_id={best_id} "
          f"texto={model.decode([best_id])!r} "
          f"num_validos={len(valid_ids)}")

    texto = model.decode([best_id])
    valor = (texto.strip() == "true")
    return valor, [best_id]


def generate_number(
    model: Small_LLM_Model,
    current_input: list[int],
    vocab: dict[str, int],
    parameter_name: str,
    es_ultimo: bool,
    i: int,
) -> tuple[float, list[int]]:
    """Generate a numeric argument value using constrained decoding.

    Args:
        model: The loaded language model.
        current_input: Token ids generated so far (context + prior values).
        vocab: The model's vocabulary, mapping token text to token id.
        es_ultimo: Whether this is the last parameter (terminator is '}'
            instead of ',').

    Returns:
        tuple[float, list[int]]: The parsed numeric value, and the list of
            token ids generated for it (terminator token excluded).
    """
    # parameter_line = f'"{parameter_name}":'
    ordinal = "first" if i == 0 else "second" if i == 1 else f"{i+1}th"
    parameter_line = (
        f'(use the {ordinal} number mentioned in the question) '
        f'"{parameter_name}":'
    )
    # parameter_line = (
    #    f'(use the number from the question that has not been used yet) '
    #    f'"{parameter_name}":'
    # )
    ids_extra = model.encode(parameter_line)[0].tolist()
    current_input = current_input + ids_extra
    digit_ids = get_digit_token_ids(vocab)
    terminator_char = "}" if es_ultimo else ","
    terminator_ids = get_terminator_token_ids(vocab, terminator_char)

    generated_ids: list[int] = []
    digits_so_far = ""
    has_digit = False
    intentos = 0

    while True:
        intentos += 1
        if intentos > 20:
            print("    [debug] ¡Demasiados intentos, abortando bucle!")
            break
        logits = model.get_logits_from_input_ids(current_input + generated_ids)
        if not has_digit:
            valid_ids = set(digit_ids.values())
        else:
            valid_ids = set(digit_ids.values()) | set(terminator_ids.values())

        best_id = max(valid_ids, key=lambda x: logits[x])

        print(f"    [debug] best_id={best_id} "
              f"texto={model.decode([best_id])!r} "
              f"es_terminador={best_id in terminator_ids.values()} "
              f"num_validos={len(valid_ids)}")
        # print(f"    [debug] parameter_line = {parameter_line!r}")
        if best_id in terminator_ids.values():
            break

        generated_ids.append(best_id)
        digits_so_far += model.decode([best_id])
        has_digit = True

    return float(digits_so_far), generated_ids


def generate_args(
    model: Small_LLM_Model,
    best_function: dict,
    prompt_item: str,
    vocab: dict[str, int],
) -> dict:
    """Generate the args dict for the chosen function,
       one parameter at a time."""
    numeros = re.findall(r'-?\d+\.?\d*', prompt_item)
    contexto_base = (
        f'User question: "{prompt_item}"\n'
        f'Function: {best_function["name"]} — {best_function["description"]}\n'
        f'Extract the argument values directly from the user question above.\n'
        f'in the order they appear. Each parameter must use a different value '
        f'from the question.\n'
        f'Numbers mentioned in the question, in order: {", ".join(numeros)}.\n'
        f'Generate a JSON object with the argment values.\n'
        f'{{'
    )
    current_input = model.encode(contexto_base)[0].tolist()

    parameters = best_function["parameters"]
    nombres_parametros = list(parameters.keys())
    args: dict = {}

    for i, nombre in enumerate(nombres_parametros):
        tipo = parameters[nombre]["type"]
        es_ultimo = (i == len(nombres_parametros) - 1)

        if tipo == "number":
            valor, tokens_generados = generate_number(
                model, current_input, vocab, nombre, es_ultimo, i
            )
        elif tipo == "boolean":
            valor, tokens_generados = generate_boolean(
                model, current_input, vocab, nombre, i
            )
        elif tipo == "string":
            valor, tokens_generados = generate_string(
                model, current_input, vocab, nombre, i
            )
        else:
            raise ValueError(f"Unsupported parameter type: {tipo!r}")

        args[nombre] = valor
        current_input = current_input + tokens_generados

    return args
