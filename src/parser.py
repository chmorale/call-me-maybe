import json
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing import Any, TypeVar, Callable, cast

T = TypeVar("T", bound=list[Any])


class PromptItem(BaseModel):
    prompt: str


PromptElement = PromptItem | str
PromptListValidator = TypeAdapter(list[PromptElement])


def gest_error(
        func: Callable[[list[dict[str, Any]]], T]
        ) -> Callable[[str], T]:
    def wrapper(file_path: str) -> T:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)

            return func(raw_data)

        except FileNotFoundError:
            print(f"Error: input file {file_path} not found")
            return cast(T, [])
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format file {file_path}")
            return cast(T, [])
        except ValidationError:
            print(f"Error: Json file {file_path} does not match the expected "
                  "schema")
            return cast(T, [])
        except Exception as e:
            print(f"Error: Unexpected Error processing file {file_path}: {e}")
            return cast(T, [])
    return wrapper


@gest_error
def prompt_parser(prompt_str: list[dict[str, Any]]) -> list[str]:
    valid_prompts = PromptListValidator.validate_python(prompt_str)

    clean_prompts = [
        elem.prompt if isinstance(elem, PromptItem) else elem
        for elem in valid_prompts
    ]

    return clean_prompts


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]
    returns: dict[str, Any]


FunctionListValidator = TypeAdapter(list[FunctionDefinition])


@gest_error
def function_parser(
        function_str: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid_functions = FunctionListValidator.validate_python(function_str)

    clean_functions = [
        elem.model_dump() if isinstance(elem, FunctionDefinition) else elem
        for elem in valid_functions
    ]

    return clean_functions
