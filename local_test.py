#!/usr/bin/env python3
import sys
import yaml
import importlib


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
    class_name = cfg.get("method")
    if not image_path:
        print("Error: 'image_path' must be specified in args.yml", file=sys.stderr)
        sys.exit(1)
    if not class_name:
        print("Error: 'method' must be specified in args.yml", file=sys.stderr)
        sys.exit(1)

    threshold = cfg.get("threshold")

    # Dynamically load and instantiate matcher class
    module = importlib.import_module('src')
    MatcherClass = getattr(module, class_name)
    matcher = MatcherClass(threshold=threshold)

    result = matcher.match(image_path)

    print(f"Best matching swatch: {result}")

if __name__ == "__main__":
    main()
