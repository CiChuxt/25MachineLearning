import inspect

from sklearn.manifold import TSNE

from common import common_arg_parser, print_result, run_reducer


def build_tsne(perplexity: float, random_state: int) -> TSNE:
    kwargs = {
        "n_components": 2,
        "perplexity": perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": random_state,
    }
    params = inspect.signature(TSNE).parameters
    if "max_iter" in params:
        kwargs["max_iter"] = 1000
    else:
        kwargs["n_iter"] = 1000
    return TSNE(**kwargs)


def main() -> None:
    parser = common_arg_parser("Run t-SNE dimensionality reduction.")
    args = parser.parse_args()

    result = run_reducer(
        "t-SNE",
        lambda: build_tsne(args.perplexity, args.random_state),
        dataset=args.dataset,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    print_result(result)


if __name__ == "__main__":
    main()

