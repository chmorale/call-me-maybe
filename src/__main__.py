import argparse
import json
import sys
import os
from .parser import prompt_parser, function_parser
from .select_function import select_function


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for input and output paths.

    Returns: argparse.Namespace: Parsed arguments with input,
        functions_definition, and output attributes.
    """
    parser = argparse.ArgumentParser(
        description="Translate natural language prompts into function calls."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the JSON file containing natural language prompts "
             "(default: data/input/function_calling_tests.json)",
    )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the JSON file containing functions definition "
             "(default: data/input/functions_definition.json)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path to the output file (default: "
             "data/output/function_calling_results.json)",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point of the program."""
    args = parse_args()

    prompt_list = prompt_parser(args.input)
    function_list = function_parser(args.functions_definition)

    if not prompt_list:
        print("No valid prompts to process. Exiting.")
        return
    if not function_list:
        print("No valid functions available. Exiting.")
        return

    try:
        results = select_function(prompt_list, function_list)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)

    output_dir = os.path.dirname(args.output)
    try:
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"Error: could not write output file {args.output!r}: {e}")
        sys.exit(1)
    # os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # with open(args.output, "w", encoding="utf-8") as f:
    #    json.dump(results, f, indent=2, ensure_ascii=False)
    # print(best_function["name"])
    # return {"<FASE_NO_IMPLEMENTADA_TODAVIA>"}


if __name__ == "__main__":
    main()
