# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    call_me_maybe.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: chmorale <chmorale@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2026/07/26 12:27:38 by chmorale          #+#    #+#              #
#    Updated: 2026/07/26 12:27:51 by chmorale         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
import json
from parser import prompt_parser, function_parser

def main() -> None:
	if len(sys.argv) != 3:
		print("Error: only funciones.json y datos.json required")
		sys.exit(1)
	
	file1 = sys.argv[1]
	file2 = sys.argv[2]

	try:
		with open(sys.argv[1], "r", encoding="utf-8") as f:
			raw_data = json.load(f)
			
		first_item = raw_data[0] if raw_data else {}
		
		if isinstance(first_item, dict) and ("parameters" in first_item or
			"returns" in first_item):
			function_file = file1
			prompt_file = file2
		else:
			function_file = file2
			prompt_file = file1
		
	except Exception as e:
		print("Error: file not reachable, aborting process")
		sys.exit(1)
	
	prompt_list = prompt_parser(prompt_file)
	for prompt_item in prompt_list:
		print (prompt_item)
	function_list = function_parser(function_file)
	for function_item in function_list:
		print (function_item)
    	
    	
if __name__ == "__main__":
	main()
