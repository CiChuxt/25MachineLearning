from sklearn.manifold import Isomap

from common import common_arg_parser, print_result, run_reducer


def main() -> None:
    parser = common_arg_parser("Run Isomap dimensionality reduction.")
    args = parser.parse_args()

    result = run_reducer(
        "Isomap",
        lambda: Isomap(n_neighbors=args.n_neighbors, n_components=2),
        dataset=args.dataset,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    print_result(result)


if __name__ == "__main__":
    main()

