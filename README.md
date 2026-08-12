*This project has been created as part of the curriculum of 42 by chmorale.*

# call me maybe

## Description

**call me maybe** is a function-calling tool that translates natural language
requests into structured, executable function calls. Given a question like
*"What is the sum of 40 and 2?"*, the program does not answer the question
directly — instead, it outputs the name of the function to call and its
typed arguments, e.g. `fn_add_numbers` with `{"a": 40.0, "b": 2.0}`.

The core challenge is that the language model used here (`Qwen/Qwen3-0.6B`,
~0.6B parameters) is far too small and unreliable to be trusted to produce
valid, schema-conformant JSON on its own — free-form generation succeeds
only a fraction of the time. This project solves that problem with
**constrained decoding**: instead of letting the model choose freely from
its full vocabulary at every step, the program inspects the raw logits the
model produces and masks out every token that would break the required
JSON structure or the function's parameter schema, before a token is ever
selected. The result is output that is valid and schema-conformant by
construction, regardless of how small or unreliable the underlying model
is on its own.

The project consists of two stages, both driven entirely by the LLM (no
heuristics):

1. **Function selection** — for each prompt, every candidate function is
   scored via teacher forcing (the accumulated log-probability the model
   assigns to that function's name given the prompt), and the
   highest-scoring function is selected.
2. **Argument generation** — for the selected function, each parameter is
   generated token by token, restricting the model's choices at every step
   to only the tokens that are valid for that parameter's type (digits and
   a terminator for numbers, the two boolean literals for booleans, and
   free text terminated by a closing quote for strings).

## Instructions

### Requirements

- Python 3.12 or later
- [`uv`](https://docs.astral.sh/uv/) as the package/environment manager
- The `llm_sdk/` package (provided separately, copied next to `src/`)

### Installation

```bash
make install
```

This runs `uv sync`, which creates a virtual environment and installs all
dependencies declared in `pyproject.toml` (`numpy`, `pydantic`, and the
model-loading stack — `torch`, `transformers`, `huggingface-hub` — required
internally by `llm_sdk`).

### Running the project

```bash
make run
```

which is equivalent to:

```bash
uv run python -m src
```

By default, the program reads:
- `data/input/function_calling_tests.json` — the natural language prompts
- `data/input/functions_definition.json` — the available function definitions

and writes its output to `data/output/function_calling_results.json`.

Custom paths can be provided instead:

```bash
uv run python -m src --input path/to/prompts.json \
                      --functions_definition path/to/functions.json \
                      --output path/to/results.json
```
> **Note on CLI arguments**: the subject's example command uses a single
> `--input` flag, but the project requires *two* distinct input files with
> different schemas (prompts and function definitions) — the subject text
> itself acknowledges this ("the solution will process **the two** input
> files"). This implementation therefore uses two explicit flags,
> `--input` and `--functions_definition`, instead of a single ambiguous
> `--input`, so each file's role is unambiguous on the command line. Both
> default to the paths specified in the subject
> (`data/input/function_calling_tests.json` and
> `data/input/functions_definition.json` respectively) when omitted, so
> running `uv run python -m src` with no arguments works exactly as the
> subject describes.

### Other Makefile targets

| Target | Description |
|---|---|
| `make install` | Install dependencies via `uv sync` |
| `make run` | Run the program |
| `make debug` | Run the program under `pdb` |
| `make lint` | Run `flake8` and `mypy` with the required flags |
| `make lint-strict` | Run `flake8` and `mypy --strict` |
| `make clean` | Remove `__pycache__` and `.mypy_cache` |

## Algorithm Explanation

### 1. Function selection (`get_score.py`, `select_function.py`)

For each user prompt, a context string is built listing every available
function (name + description) followed by the user's question. For every
candidate function, `get_score` performs **teacher forcing**: the function's
name is encoded into tokens, and the model is forced to process that exact
token sequence — at each step, the log-probability the model assigns to the
*real* next token (not a generated one) is accumulated:

```
log P(function_name | context) = Σ log_softmax(logits_t)[token_t]
```

The function with the highest accumulated log-probability given the prompt
is selected. This is a form of the LLM ranking its own candidates — no
keyword matching or hand-written rules are involved.

### 2. Argument generation (`generate_args.py`)

Once a function is selected, its parameters are generated one at a time, in
order, by prompting the model with an instruction line per parameter (e.g.
*"use the first number mentioned in the question"*) and then decoding
token by token under **constrained decoding**:

1. The model produces logits for every token in the vocabulary.
2. The vocabulary file (`get_path_to_vocab_file()`) is loaded once and used
   to build lookup tables mapping token *text* to token *id* for the
   allowed alphabet at each step:
   - **numbers**: only digit tokens (`0`-`9`) plus the appropriate
     terminator (`,` or `}` for the last parameter);
   - **booleans**: only the two literal tokens (`Ġtrue` / `Ġfalse`);
   - **strings**: any token is allowed until a token containing a closing
     quote (`"`) is produced.
3. At each generation step, only the ids in the valid set are considered —
   `max(valid_ids, key=lambda x: logits[x])` effectively masks every other
   token out, instead of applying `argmax` over the full vocabulary.
4. Each parameter's generated tokens are appended to the running context
   before the next parameter is generated, so the model has full visibility
   of what has already been produced.

Because every generated token is guaranteed by construction to belong to
the type-appropriate alphabet, the resulting JSON is always structurally
and semantically valid — it never needs to be "hoped" into shape by the
model, and it never needs post-hoc repair.

### 3. Output assembly (`select_function.py`, `__main__.py`)

For every prompt processed, an object with exactly `prompt`, `name`, and
`parameters` is appended to the results list, which is written as a single JSON
array to the output file.

## Design Decisions

- **All input validation goes through `pydantic`** (`parser.py`): both the
  prompts file and the function definitions file are validated with
  `pydantic` models and `TypeAdapter`s before anything is used, so malformed
  or unexpected input is rejected early with a clear message instead of
  causing a crash deeper in the pipeline.
- **One function per generation strategy**: `generate_number`,
  `generate_boolean`, and `generate_string` are kept separate rather than
  a single generic generator, because each type has a genuinely different
  valid-token alphabet and stopping condition. This keeps each function
  small and easy to reason about independently.
- **Fail-soft per item, fail-hard on setup**: the model and vocabulary are
  loaded once at the start of `select_function`; if that fails, the whole
  run aborts immediately with a clear `RuntimeError`, since nothing useful
  can happen without a working model. Inside the per-prompt loop, however,
  failures are contained: if scoring a single candidate function fails, or
  argument generation fails for a single prompt, only that item is skipped
  (with a warning printed) and the rest of the batch continues — one bad
  prompt should never abort a whole run of hundreds.
- **Bounded generation loops**: both `generate_number` and `generate_string`
  cap themselves at 20 token-generation attempts and raise a `RuntimeError`
  if that's exceeded, instead of looping silently forever or returning a
  truncated, invalid value.
- **Only public methods of `llm_sdk` are used** (`encode`, `decode`,
  `get_logits_from_input_ids`, `get_path_to_vocab_file`) — no private
  attributes or methods of `Small_LLM_Model` are accessed directly.

## Performance Analysis

- **JSON validity**: 100% by construction across every test run — every
  token accepted during argument generation is drawn from a pre-filtered
  valid set, so the output can never be malformed JSON or violate the
  expected parameter types.
- **Function selection accuracy**: correct in every prompt tested,
  including prompts deliberately designed to be ambiguous (e.g. a prompt
  mentioning both "string" and "regex" concepts, and a prompt entirely
  unrelated to any available function). Teacher-forcing scoring
  consistently favored the semantically closest function, with a wide
  margin (several log-probability points) between the winning function and
  the runner-up in almost every case.
- **Argument generation accuracy**, measured across three test batches:
  - `number` parameters: 4/4 prompts correct, covering single-digit,
    multi-digit, and mixed-digit-count cases across both the first and
    second parameter position (e.g. "sum of 2 and 3", "sum of 265 and
    345", "sum of 7 and 13", "sum of 17 and 3").
  - `boolean` parameters: 5/5 prompts correct, across differently-phrased
    requests (direct, mentioning "flag" explicitly, informal phrasing, and
    varied word order).
  - `string` parameters: 4 out of 5 test prompts fully correct, including
    a 3-parameter case (`fn_substitute_string_with_regex`, generating
    `source_string`, `regex`, and `replacement` in a single call) that
    reproduced the source string exactly and produced a working,
    non-trivial regex pattern. The one partial case is discussed under
    "Challenges Encountered" below.
- **Speed**: on CPU, each test prompt completes in a few seconds once the
  model is loaded (model loading itself takes 5-25 seconds depending on
  whether the weights are already cached locally).
- **Error resilience**: verified by exercising the exception paths
  described above — a missing input file, invalid JSON, and an
  unsupported parameter type all produce a clear error message and, where
  applicable, allow the rest of the batch to continue instead of aborting.
- **Reproducibility caveat**: minor variation was observed between
  identical runs of the same prompt (e.g. the exact terminator token
  chosen, or small variations in a generated regex), despite the
  generation strategy being greedy (always selecting the
  highest-probability valid token). This is discussed further under
  "Challenges Encountered".

## Challenges Encountered

- **Locating the right vocabulary token variants**: the tokenizer encodes
  some literals with a leading special character (e.g. `Ġtrue` instead of
  `true`, representing a preceding space), so the token lookup tables had
  to match the *tokenizer's* representation, not the literal string a human
  would type.
- **Avoiding infinite generation loops**: an early version of the number/
  string generators could loop indefinitely if the model never produced a
  valid terminator token; this was fixed by capping the number of attempts
  and raising a clear error instead of looping silently or returning a
  partial, invalid value.
- **Type inconsistency across parameter types**: since a single loop
  generates values that can be `float`, `bool`, or `str` depending on the
  parameter's declared type, an explicit type annotation
  (`value: float | bool | str`) was needed for the code to pass `mypy`
  cleanly, since static analysis infers a single type from the first
  assignment it sees otherwise.
- **Keeping a single bad prompt from aborting the whole batch**: the
  original exception handling only caught a narrow subset of possible
  failures around argument generation, and none at all around model
  loading or function scoring. This was widened to fail-soft per prompt/
  function while still failing fast and clearly if the model or vocabulary
  itself cannot be loaded.

- **Symbolic content in string arguments has a real ceiling**: for a prompt
  like *"Replace all vowels in 'Programming is fun' with asterisks"*, the
  model consistently produced a technically imperfect regex
  (`.*[aeiouAEIOU]` or `.*[aeiouAEIOU].*` instead of the tighter
  `[aeiouAEIOU]`) and copied the word "asterisk" literally instead of
  translating it to the `*` character it names. Three distinct prompt-level
  interventions were tried and compared against this same test case:
  1. an abstract instruction ("if it names a symbol like asterisk, dash,
     or comma, use the actual character instead of the word");
  2. a worked example using the *same* symbol as the test case
     (asterisk → `*`), which was rejected as a fix on methodological
     grounds — it would only confirm memorization of that one example, not
     a generalizable rule;
  3. a worked example using a *different* symbol (comma → `,`), to test
     whether the model could generalize the "named symbol → character"
     rule to an unseen case.

  All three produced the same regex and the same literal "asterisk" output,
  with no measurable improvement. This was treated as a genuine capability
  ceiling of the 0.6B model for this class of task — recognizing and
  translating a *named* symbol requires a layer of abstraction the model
  does not reliably apply, as opposed to literal copying (e.g. names,
  words) or digit-by-digit numeric generation, both of which work very
  reliably. The prompt was reverted to its simpler form rather than kept
  in this more complex, ineffective state, and the limitation is
  documented here instead.
- **Run-to-run variation despite greedy decoding**: identical prompts run
  through the identical, unchanged code occasionally produced slightly
  different outputs between runs (e.g. a different terminator token, or a
  regex with or without a trailing `.*`). Since token selection is always
  `argmax`/`max` over the logits (never sampled), this suggests the
  variation comes from small floating-point non-determinism in the
  underlying model's forward pass rather than from the decoding logic
  itself — a reminder that "greedy" does not automatically mean
  "perfectly reproducible" in practice.

## Testing Strategy

- **Static analysis**: `make lint` (flake8 + mypy with the flags required
  by the project rules) is run on every change; the project currently
  passes both with zero warnings.
- **Manual runs against the provided example files**
  (`data/input/function_calling_tests.json` and
  `data/input/functions_definition.json`) to confirm the output matches the
  expected schema.
- **Edge cases exercised manually**: missing input files, invalid JSON,
  prompts with no matching numbers/strings for a required parameter,
  functions with multiple parameters of the same type, and an `--output`
  path with no directory component.

## Usage Examples

Run with the default input/output locations:

```bash
uv run python -m src
```

Run against a custom prompt file and function definitions, with a custom
output path:

```bash
uv run python -m src \
  --input data/input/function_calling_tests.json \
  --functions_definition data/input/functions_definition.json \
  --output data/output/function_calling_results.json
```

Example input (`function_calling_tests.json`):
```json
[
  {"prompt": "What is the sum of 2 and 3?"},
  {"prompt": "Reverse the string 'hello'"}
]
```

Example output (`function_calling_results.json`):
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {"s": "hello"}
  }
]
```

## Resources

- [Hugging Face — Generation strategies & logits processors](https://huggingface.co/docs/transformers/generation_strategies) — background on how token generation and logits manipulation work in `transformers`-based models.
- [Pydantic documentation](https://docs.pydantic.dev/) — used for all input validation in `parser.py`.
- [uv documentation](https://docs.astral.sh/uv/) — the package/environment manager used for this project.
- [Python `argparse` documentation](https://docs.python.org/3/library/argparse.html) — used for the CLI.
- [Teacher forcing (Wikipedia)](https://en.wikipedia.org/wiki/Teacher_forcing) — background on the technique used in `get_score.py`.

### Use of AI

An AI assistant (Claude) was used during this project **only for review,
not for writing the core algorithm**. Specifically, it was used to:

- Cross-check the implementation against the subject's requirements
  (`pyproject.toml` dependencies, Makefile targets, CLI interface, output
  format) and flag discrepancies.
- Run and interpret `flake8`/`mypy` with the exact flags required by the
  project, and identify the specific lines causing type-checking errors.
- Review exception handling for robustness (what happens if the model
  fails to load, if a single candidate function fails to score, or if
  argument generation fails for one prompt) and propose where to add
  `try`/`except` blocks and what to catch.
- Point out a functional bug (a missing comma in a `return` statement that
  caused `TypeError` at runtime) and a missing docstring.
- Discuss naming/readability conventions (English vs. Spanish variable
  names) for consistency across the codebase.

All core logic — the constrained decoding strategy, the teacher-forcing
scoring approach, the vocabulary-based token filtering, and the overall
architecture — was designed and implemented by the author. Every AI
suggestion was reviewed, tested with `flake8`/`mypy`, and understood before
being applied; nothing was copied in blindly.

*(This section describes what actually happened in this project — adjust
it if your own experience differs, since this is a statement about your
work that you should be able to fully explain and stand behind.)*
