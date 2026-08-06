import numpy as np
from llm_sdk import Small_LLM_Model


def get_score(model: Small_LLM_Model,
              prompt: str,
              function_item: dict) -> float:
    """Score how likely a candidate function is, given a context prompt.

    Uses teacher forcing: forces the model to process the function name
    token by token and accumulates the log-probability the model assigns
    to each real token, instead of letting it generate freely.

    Args:
        model: The loaded language model used to compute token logits.
        prompt: The full context text (available functions + user prompt).
        function_item: The candidate function definition, must contain
            at least a "name" key.

    Returns:
        float: The accumulated log-probability of the function name given
            the context. Higher (closer to 0) means more likely.
    """
    value = 0.0
    input_ids = model.encode(prompt)[0].tolist()
    current_input = input_ids.copy()

    function_name = function_item["name"]

    function_tokens = model.encode(function_name)[0].tolist()
    for token_id in function_tokens:
        logits = model.get_logits_from_input_ids(current_input)
        logit_token = logits[token_id]
        current_input.append(token_id)
        log_probs = logit_token - np.log(np.sum(np.exp(logits)))
        value += log_probs
        # token_text = model.decode([token_id])
        # print(f"  token: {token_text!r}  (id={token_id})  "
        #       f"log_prob={log_probs:.4f}  acumulado={value:.4f}")
    return value
