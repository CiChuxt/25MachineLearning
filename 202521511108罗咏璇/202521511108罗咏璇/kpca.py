from sklearn.decomposition import KernelPCA

from common import common_arg_parser, print_result, run_reducer


def main() -> None:
    parser = common_arg_parser("Run Kernel PCA dimensionality reduction.")
    args = parser.parse_args()

    result = run_reducer(
        "KPCA",
        lambda: KernelPCA(
            n_components=2,
            kernel="rbf",
            gamma=args.gamma,
            fit_inverse_transform=False,
        ),
        dataset=args.dataset,
        sample_size=args.sample_size,
        random_state=args.random_state,
    )
    print_result(result)


if __name__ == "__main__":
    main()

