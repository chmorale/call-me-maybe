import json
from llm_sdk import Small_LLM_Model
from .get_score import get_score
from .generate_args import generate_args


def get_context(prompt_item, function_list) -> str:
    """Build the context text listing available functions and the user prompt.

    Args:
        prompt_item: The natural language prompt from the user.
        function_list: List of available function definitions.

    Returns:
        str: A single text block with all functions and the prompt, ready
            to be tokenized by the model.
    """
    # lineas_funciones = []
    lineas_funciones = ["Select the best function to solve this prompt: "]
    for func in function_list:
        linea = f"- {func['name']}: {func['description']}"
        lineas_funciones.append(linea)

    bloque_funciones = "\n".join(lineas_funciones)

    contexto = (
        f"Available functions:\n{bloque_funciones}\n\n"
        f"User question: \"{prompt_item}\"\n"
        f"Function to call:"
    )

    return contexto


def select_function(
        prompt_list: list[str],
        function_list: list[dict],
) -> list[dict]:

    model = Small_LLM_Model()
    with open(model.get_path_to_vocab_file(), "r", encoding="utf-8") as f:
        vocab = json.load(f)
    results = []
    for prompt_item in prompt_list:
        best_function: dict | None = None
        # best_function: dict
        best_score = float('-inf')
        contexto = get_context(prompt_item, function_list)
        # print("============")
        # print(prompt_item)
        # print("posibles funciones:")
        for function_item in function_list:
            # print(function_item)
            score = get_score(model, contexto, function_item)
            if score > best_score:
                best_score = score
                best_function = function_item
        if best_function is not None:
            print(f"Prompt: {prompt_item}")
            print(f"Función elegida: {best_function['name']}  "
                  f"(score={best_score:.4f})")
            print("============")
            try:
                args = generate_args(model, best_function, prompt_item, vocab)
            except ValueError as e:
                print(f"Error generating args for prompt {prompt_item!r}: {e}")
                continue

            results.append({
                "prompt": prompt_item,
                "fn_name": best_function['name'],
                "args": args,
            })

    return results
    # return best_function
