import sys

from sample_factory.enjoy import enjoy
from sf_examples.robosuite.train_robosuite import parse_robosuite_args, register_robosuite_components


def main():
    """Script entry point."""
    register_robosuite_components()
    cfg = parse_robosuite_args(evaluation=True)
    status = enjoy(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
