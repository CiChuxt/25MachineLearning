from __future__ import annotations

from sklearn.decomposition import KernelPCA, PCA
from sklearn.manifold import Isomap, LocallyLinearEmbedding

from common import common_arg_parser, print_result, run_reducer, save_results
from mds import build_mds
from tsne import build_tsne
from umap_method import build_umap


def main() -> None:
    parser = common_arg_parser("Run all dimensionality reduction methods.")
    args = parser.parse_args()

    methods = [
        (
            "PCA",
            lambda: PCA(n_components=2, random_state=args.random_state),
        ),
        (
            "MDS",
            lambda: build_mds(args.random_state),
        ),
        (
            "KPCA",
            lambda: KernelPCA(
                n_components=2,
                kernel="rbf",
                gamma=args.gamma,
                fit_inverse_transform=False,
            ),
        ),
        (
            "Isomap",
            lambda: Isomap(n_neighbors=args.n_neighbors, n_components=2),
        ),
        (
            "LLE",
            lambda: LocallyLinearEmbedding(
                n_neighbors=args.n_neighbors,
                n_components=2,
                method="standard",
                eigen_solver="arpack",
                random_state=args.random_state,
            ),
        ),
        (
            "t-SNE",
            lambda: build_tsne(args.perplexity, args.random_state),
        ),
        (
            "UMAP",
            lambda: build_umap(args.n_neighbors, args.min_dist, args.random_state),
        ),
    ]

    rows = []
    for method_name, factory in methods:
        print(f"\nRunning {method_name}...")
        try:
            result = run_reducer(
                method_name,
                factory,
                dataset=args.dataset,
                sample_size=args.sample_size,
                random_state=args.random_state,
            )
            print_result(result)
            rows.append(result)
        except Exception as exc:
            print(f"{method_name} failed: {exc}")
            rows.append(
                {
                    "method": method_name,
                    "dataset": args.dataset,
                    "n_samples": args.sample_size,
                    "n_features": "",
                    "figure": "",
                    "silhouette_score": "",
                    "trustworthiness": "",
                    "sse": "",
                    "runtime_seconds": "",
                    "error": str(exc),
                }
            )

    save_results(rows)
    print("\nAll available methods finished.")
    print("Figures saved to: experiment_results/figures/")
    print("Metrics saved to: experiment_results/results/metrics.csv")
    print("Summary saved to: experiment_results/results/summary.md")


if __name__ == "__main__":
    main()
