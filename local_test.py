#!/usr/bin/env python3
import sys
import yaml
from src import SwatchMatcher

def main():
    # Load arguments from YAML
    try:
        with open("local_test_params.yml", "r") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print("Error: args.yml not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading args.yml: {e}", file=sys.stderr)
        sys.exit(1)

    image_path = cfg.get("image_path")
    if not image_path:
        print("Error: 'image_path' must be specified in args.yml", file=sys.stderr)
        sys.exit(1)

    threshold = cfg.get("threshold")

    # Instantiate and run matcher
    matcher = SwatchMatcher(threshold=threshold)
    try:
        result = matcher.match(image_path)
    except Exception as e:
        print(f"Error during matching: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Best matching swatch: {result}")
