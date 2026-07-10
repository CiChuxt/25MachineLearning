from common import common_arg_parser, print_result, run_reducer


def build_umap(n_neighbors: int, min_dist: float, random_state: int):
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError(
            "UMAP requires the optional package `umap-learn`. "
            "Install it with: pip install umap-learn"
        ) from exc

    return UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=random_state,
    )


def main() -> None:
    parser = common_arg_parser("Run UMAP dimensionality reduction.")
    args = parser.parse_args()

    result = run_reducer(
        "UMAP",
        lambda: build_umap(args.n_neighbors, args.min_dist, args.random_state),
        dataset=args.dataset,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    print_result(result)


if __name__ == "__main__":
    main()

