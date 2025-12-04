#!/usr/bin/env python3
"""
Add dummy tokens to a Hugging Face tokenizer until its total vocabulary size matches a target size.

Usage example:
  python add_dummy_tokens.py \
	--input /path/to/tokenizer_or_model_dir \
	--target-size 50288 \
	--output /path/to/output_tokenizer_dir

Notes:
	- This updates only the tokenizer files. If you plan to use the tokenizer with a model,
	remember to resize the model embeddings accordingly (e.g., model.resize_token_embeddings(len(tokenizer))).
  - If current size exceeds the target, this script will error out (no shrinking by default).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Set

try:
	from transformers import AutoTokenizer, PreTrainedTokenizerBase, PreTrainedTokenizerFast
except Exception as _e:  # pragma: no cover - runtime import guard
	print(
		"Missing dependency: transformers. Install with 'pip install transformers tokenizers'.",
		file=sys.stderr,
	)
	raise


def _load_tokenizer(path: str) -> PreTrainedTokenizerBase:
	"""Load a tokenizer from a directory or a single tokenizer.json file.

	Prefers fast tokenizers when available.
	"""
	if os.path.isdir(path):
		# Standard path (contains tokenizer.json and friends)
		tok = AutoTokenizer.from_pretrained(path, use_fast=True)
		return tok

	# Fallback: single tokenizer.json file
	if os.path.isfile(path) and path.endswith(".json"):
		return PreTrainedTokenizerFast(tokenizer_file=path)

	# As a last resort, try from_pretrained directly (e.g., HF hub ID)
	return AutoTokenizer.from_pretrained(path, use_fast=True)


def _current_size(tokenizer: PreTrainedTokenizerBase) -> int:
	"""Return total known token count (base vocab + added tokens)."""
	try:
		# In HF, len(tokenizer) counts base + added tokens
		return len(tokenizer)
	except Exception:
		# Fallback to get_vocab
		return len(tokenizer.get_vocab())


def _generate_dummy_tokens(
	existing: Set[str],
	count: int,
	template: str = "<unused_{}>",
	start_index: int = 0,
) -> List[str]:
	"""Generate a list of unique dummy tokens not present in `existing`.

	Ensures no collisions with existing tokens by incrementing the index until
	enough unique tokens are produced.
	"""
	out: List[str] = []
	i = start_index
	while len(out) < count:
		candidate = template.format(i)
		if candidate not in existing and candidate not in out:
			out.append(candidate)
		i += 1
	return out


def add_dummy_tokens(
	input_path: str,
	target_size: int,
	output_dir: str,
	dummy_template: str = "<unused_{}>",
	start_index: int = 0,
	write_report: bool = True,
) -> None:
	tokenizer = _load_tokenizer(input_path)

	current = _current_size(tokenizer)
	if current == target_size:
		os.makedirs(output_dir, exist_ok=True)
		tokenizer.save_pretrained(output_dir)
		print(f"Tokenizer already at target size {target_size}. Saved as-is to: {output_dir}")
		return

	if current > target_size:
		raise ValueError(
			f"Current tokenizer size ({current}) is larger than target size ({target_size}). "
			"Shrinking/removal is not supported by this script."
		)

	to_add = target_size - current
	vocab: Set[str] = set(tokenizer.get_vocab().keys())

	# Generate candidates and attempt to add; ensure we actually add the required number.
	new_tokens_total: List[str] = []
	attempt_index = start_index

	while to_add > 0:
		# Generate exactly the remaining count to avoid overshooting the target
		gen_count = to_add
		candidates = _generate_dummy_tokens(vocab, gen_count, dummy_template, attempt_index)
		# Only add up to the remaining number of tokens
		actually_added = tokenizer.add_tokens(candidates[:to_add])

		if actually_added <= 0:
			# Extremely unlikely unless tokenizer backend rejects tokens
			raise RuntimeError(
				"Failed to add any new tokens; the tokenizer backend may not support additions."
			)

		# Update structures
		new_tokens_total.extend(candidates[:actually_added])
		vocab.update(candidates[:actually_added])
		to_add -= actually_added
		attempt_index += gen_count

	# Save the updated tokenizer
	os.makedirs(output_dir, exist_ok=True)
	tokenizer.save_pretrained(output_dir)

	if write_report:
		report_path = os.path.join(output_dir, "added_dummy_tokens.json")
		with open(report_path, "w", encoding="utf-8") as f:
			json.dump({"added": new_tokens_total}, f, ensure_ascii=False, indent=2)

	final_size = _current_size(tokenizer)
	print(
		f"Added {len(new_tokens_total)} tokens. Final tokenizer size: {final_size}. Saved to: {output_dir}"
	)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description=__doc__)
	p.add_argument(
		"--input",
		required=True,
		help="Path to tokenizer directory (preferred), a tokenizer.json file, or a model repo ID.",
	)
	p.add_argument(
		"--target-size",
		type=int,
		required=True,
		help="Desired total vocabulary size (base + added tokens).",
	)
	p.add_argument(
		"--output",
		required=True,
		help="Directory to write the updated tokenizer files.",
	)
	p.add_argument(
		"--dummy-template",
		default="<unused_{}>",
		help="Format template for generated dummy tokens, must contain '{}' (default: '<unused_{}>').",
	)
	p.add_argument(
		"--start-index",
		type=int,
		default=0,
		help="Starting index for dummy token numbering (default: 0).",
	)
	p.add_argument(
		"--no-report",
		action="store_true",
		help="Do not write added_dummy_tokens.json report.",
	)
	return p.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
	args = parse_args(argv)

	if "{}" not in args.dummy_template:
		print("--dummy-template must contain '{}' placeholder", file=sys.stderr)
		return 2

	if args.target_size <= 0:
		print("--target-size must be > 0", file=sys.stderr)
		return 2

	try:
		add_dummy_tokens(
			input_path=args.input,
			target_size=args.target_size,
			output_dir=args.output,
			dummy_template=args.dummy_template,
			start_index=args.start_index,
			write_report=not args.no_report,
		)
	except Exception as e:
		print(f"Error: {e}", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

