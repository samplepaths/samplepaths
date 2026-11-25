"""Generate the figures shown in the documentation.

Run from the repository root::

    make docs-images

or directly, optionally with another output directory::

    uv run python docs/scripts/make_images.py --output /tmp/images
"""

import argparse
from pathlib import Path

from _images.guides import make_branding, make_guides
from _images.paths import make_continuous, make_diffusion
from _images.series import make_discrete, make_multivariate, make_noise, make_timeseries
from plot_style import use_theme

from samplepaths import random

DEFAULT_OUTPUT = Path(__file__).parent.parent / "_static"


def main() -> None:
    """Generate every figure of the documentation."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="directory to write the images to (default: docs/_static)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed of the random number generator, for reproducible figures",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    if args.seed is not None:
        random.seed(value=args.seed)

    use_theme()

    make_branding(args.output)
    make_guides(args.output)
    make_continuous(args.output)
    make_diffusion(args.output)
    make_discrete(args.output)
    make_timeseries(args.output)
    make_multivariate(args.output)
    make_noise(args.output)


if __name__ == "__main__":
    main()
