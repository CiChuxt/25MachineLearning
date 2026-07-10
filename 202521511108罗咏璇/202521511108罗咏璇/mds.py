import inspect

from sklearn.manifold import MDS

from common import common_arg_parser, print_result, run_reducer


def build_mds(random_state: int) -> MDS:
    kwargs = {
        "n_components": 2,
        "random_state": random_state,
        "n_init": 4,
        "max_iter": 300,
    }
    if "normalized_stress" in inspect.signature(MDS).parameters:
        kwargs["normalized_stress"] = "auto"
    return MDS(**kwargs)


def main() -> None:
    parser = common_arg_parser("Run MDS dimensionality reduction.")
    args = parser.parse_args()

    result = run_reducer(
        "MDS",
        lambda: build_mds(args.random_state),
        dataset=args.dataset,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    print_result(result)


if __name__ == "__main__":
    main()

