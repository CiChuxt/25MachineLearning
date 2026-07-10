from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PACKAGE_ROOT / "results"
FIGURES_DIR = PACKAGE_ROOT / "figures"
REPORT_DIR = PACKAGE_ROOT / "report"
QA_DIR = PACKAGE_ROOT / "qa"

REPORT_TITLE = "基于UCI HAR智能手机人体活动识别数据的高维特征降维方法对比研究"
REPORT_FILE = REPORT_DIR / "高维传感器数据降维方法对比研究.docx"

DATASET_URLS = [
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip",
]

DATASET_PAGE = "https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones"
DATASET_ZIP = RAW_DIR / "uci_har_dataset.zip"
DATASET_FOLDER = RAW_DIR / "UCI HAR Dataset"

ACTIVITY_LABELS = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    slug: str
    supports_transform: bool
    classification_dimensions: tuple[int, ...]
    visualization_dimensions: tuple[int, ...] = (2,)


DIMENSIONALITY_METHODS = [
    MethodSpec("PCA", "pca", True, (10, 30, 50)),
    MethodSpec("MDS", "mds", False, (2,)),
    MethodSpec("Kernel PCA", "kernel_pca", True, (10, 30, 50)),
    MethodSpec("Isomap", "isomap", True, (10, 30, 50)),
    MethodSpec("t-SNE", "tsne", False, (2,)),
    MethodSpec("UMAP", "umap", True, (10, 30, 50)),
    MethodSpec("LLE", "lle", True, (10, 30, 50)),
]

REFERENCES = [
    "Anguita, D., Ghio, A., Oneto, L., Parra, X., and Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN.",
    "UCI Machine Learning Repository. Human Activity Recognition Using Smartphones Dataset. https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecognition%2Busing%2Bsmartphones",
    "Hotelling, H. (1933). Analysis of a complex of statistical variables into principal components. Journal of Educational Psychology, 24(6), 417-441.",
    "Jolliffe, I. T. (2002). Principal Component Analysis. Springer.",
    "Torgerson, W. S. (1952). Multidimensional scaling: I. Theory and method. Psychometrika, 17, 401-419.",
    "Kruskal, J. B. (1964). Multidimensional scaling by optimizing goodness of fit to a nonmetric hypothesis. Psychometrika, 29, 1-27.",
    "Scholkopf, B., Smola, A., and Muller, K. R. (1998). Nonlinear component analysis as a kernel eigenvalue problem. Neural Computation, 10(5), 1299-1319.",
    "Tenenbaum, J. B., de Silva, V., and Langford, J. C. (2000). A global geometric framework for nonlinear dimensionality reduction. Science, 290(5500), 2319-2323.",
    "Roweis, S. T., and Saul, L. K. (2000). Nonlinear dimensionality reduction by locally linear embedding. Science, 290(5500), 2323-2326.",
    "van der Maaten, L., and Hinton, G. (2008). Visualizing data using t-SNE. Journal of Machine Learning Research, 9, 2579-2605.",
    "McInnes, L., Healy, J., and Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.",
    "Belkin, M., and Niyogi, P. (2003). Laplacian Eigenmaps for dimensionality reduction and data representation. Neural Computation, 15(6), 1373-1396.",
    "Cover, T., and Hart, P. (1967). Nearest neighbor pattern classification. IEEE Transactions on Information Theory, 13(1), 21-27.",
    "Cortes, C., and Vapnik, V. (1995). Support-vector networks. Machine Learning, 20, 273-297.",
    "Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.",
    "Fawcett, T. (2006). An introduction to ROC analysis. Pattern Recognition Letters, 27(8), 861-874.",
    "Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis. Journal of Computational and Applied Mathematics, 20, 53-65.",
    "Venna, J., and Kaski, S. (2001). Neighborhood preservation in nonlinear projection methods: An experimental study. In Artificial Neural Networks - ICANN.",
]


def ensure_directories() -> None:
    for path in [RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR, REPORT_DIR, QA_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def result_path(filename: str) -> Path:
    return RESULTS_DIR / filename


def figure_path(filename: str) -> Path:
    return FIGURES_DIR / filename
