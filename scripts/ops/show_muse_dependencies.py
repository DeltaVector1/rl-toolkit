#!/usr/bin/env python3
"""Print the Python packages that the Muse Transformers override depends on."""

import importlib.metadata


for package in (
    "transformers",
    "safetensors",
    "tokenizers",
    "huggingface-hub",
    "packaging",
    "regex",
    "sentencepiece",
):
    print(package, repr(importlib.metadata.version(package)), flush=True)

if "--import-transformers" in __import__("sys").argv:
    import transformers

    print("imported-transformers", transformers.__version__)
