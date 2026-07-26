# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    parser.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: chmorale <chmorale@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/07/26 12:28:02 by chmorale          #+#    #+#              #
#    Updated: 2026/07/26 12:28:10 by chmorale         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import json
from pydantic import BaseModel, TypeAdapter

class PromptItem(BaseModel):
	prompt: str

PromptElement = PromptItem | str
PromptListValidator = TypeAdapter(list[PromptElement])


def prompt_parser(file_path: str) -> list[str]:
	try:
		with open(file_path, "r", encoding="utf-8") as f:
			raw_data = json.load(f)

		Valid_prompts = PromptListValidator.validate_python(raw_data)

		clean_prompts = [
			elem.prompt if isinstance(elem, PromptItem) else elem
			for elem in Valid_prompts
		]

		return clean_prompts
	except Exception as e:
		print(f"Error procesing file {file_path}: {e}")
		return []


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict
    returns: dict


FunctionElement = FunctionDefinition | str
FunctionListValidator = TypeAdapter(list[FunctionElement])


def function_parser(file_path: str) -> list[str]:
	try:
		with open(file_path, "r", encoding="utf-8") as f:
			raw_data = json.load(f)

		Valid_functions = FunctionListValidator.validate_python(raw_data)

		clean_functions = [
			elem.name if isinstance(elem, FunctionDefinition) else elem
			for elem in Valid_functions
		]

		return clean_functions

	except Exception as e:
		print(f"Error procesing file {file_path}: {e}")
		return []
