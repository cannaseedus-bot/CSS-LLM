#!/usr/bin/env python3
"""Placeholder conversion stub for HF -> CLIF-1 int4."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--out", default="weights/model-1b-int4.bin")
    args = parser.parse_args()

    print(f"Stub: convert {args.model} to {args.out} (implement quantization backend)")


if __name__ == "__main__":
    main()
