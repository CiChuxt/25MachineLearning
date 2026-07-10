from sklearn.manifold import LocallyLinearEmbedding

from common import common_arg_parser, print_result, run_reducer


def main() -> None:
    parser = common_arg_parser("Run LLE dimensionality reduction.")
    args = parser.parse_args()

    result = run_reducer(
        "LLE",
        lambda: LocallyLinearEmbedding(
            n_neighbors=args.n_neighbors,
            n_components=2,
            method="standard",
            eigen_solver="arpack",
            random_state=args.random_state,
        ),
        dataset=args.dataset,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    print_result(result)


if __name__ == "__main__":
    main()

