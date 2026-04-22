from itertools import product

from tsann.experiments.run_single import main as run_single


def main() -> None:
    # Placeholder grid entry point. The first milestone keeps this executable and
    # leaves the large matrix to configuration-driven expansion.
    for _ in product([0.0], [0.1]):
        run_single()


if __name__ == "__main__":
    main()
