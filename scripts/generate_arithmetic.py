#!/usr/bin/env python3
"""
Generate Arithmetic Expression Dataset

Generates corpus of arithmetic expressions with distractors for MMEP training.
Output format: JSONL with query, answer, and distractor values.

Usage:
  python scripts/generate_arithmetic.py --corpus-size 10000 --output data/arithmetic.jsonl

Contact: jessica@collectivekitty.com
"""

import argparse
import json
import random
import os
from pathlib import Path


def generate_expression(rng, max_val=100, max_ops=3):
    """Generate a random arithmetic expression and its result."""
    ops = ['+', '-', '*']
    num_ops = rng.randint(1, max_ops)

    values = [rng.randint(1, max_val) for _ in range(num_ops + 1)]
    operators = [rng.choice(ops) for _ in range(num_ops)]

    # Build expression string
    expr_parts = [str(values[0])]
    for i, op in enumerate(operators):
        expr_parts.append(f" {op} {values[i+1]}")
    expr = "".join(expr_parts)

    # Compute answer
    try:
        answer = eval(expr)
    except Exception:
        answer = values[0]

    return expr, int(answer)


def generate_distractors(answer, rng, num_distractors=3):
    """Generate plausible wrong answers."""
    distractors = set()
    attempts = 0
    while len(distractors) < num_distractors and attempts < 100:
        offset = rng.choice([-10, -5, -2, -1, 1, 2, 5, 10, 20])
        d = answer + offset
        if d != answer:
            distractors.add(d)
        attempts += 1
    return list(distractors)[:num_distractors]


def generate_dataset(size, rng, split_name="train"):
    """Generate a dataset of arithmetic expressions."""
    data = []
    for _ in range(size):
        expr, answer = generate_expression(rng)
        distractors = generate_distractors(answer, rng)
        data.append({
            "query": expr,
            "answer": answer,
            "distractors": distractors,
            "split": split_name
        })
    return data


def main():
    parser = argparse.ArgumentParser(description="Generate arithmetic dataset")
    parser.add_argument("--corpus-size", type=int, default=10000)
    parser.add_argument("--train-queries", type=int, default=5000)
    parser.add_argument("--val-queries", type=int, default=500)
    parser.add_argument("--test-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/arithmetic.jsonl")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Generating arithmetic dataset (seed={args.seed})...")

    train = generate_dataset(args.train_queries, rng, "train")
    val = generate_dataset(args.val_queries, rng, "val")
    test = generate_dataset(args.test_queries, rng, "test")

    all_data = train + val + test

    with open(output_path, 'w') as f:
        for item in all_data:
            f.write(json.dumps(item) + '\n')

    print(f"Generated {len(all_data)} examples:")
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    main()
