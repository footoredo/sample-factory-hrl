import sys

from sample_factory.enjoy import enjoy
from sf_examples.crafter_safe.train_crafter import parse_args, register_custom_components


def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
