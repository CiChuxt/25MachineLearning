from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path("data.xlsx")
FIG_DIR = Path("figures")
RANDOM_STATE = 42

FEATURE_NAMES = [
    "城区面积",
    "建成区面积",
    "人均公园绿地面积",
    "城市建设用地面积",
    "年末实有道路长度",
    "年末实有道路面积",
    "城市排水管道长度",
    "城市道路照明灯",
    "年末公共交通车辆运营数",
    "运营线路总长度",
    "每万人拥有公共交通车辆",
    "出租汽车数量",
]


def setup_plot_style():
    plt.rcParams["font.sans-serif"] = ["SimSun", "SimHei", "Microsoft YaHei"]
    plt.rcParams["axes.unicode_minus"] = False
    FIG_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_excel(DATA_FILE)
    region_col = df.columns[0]
    regions = df[region_col].astype(str)
    x = df.iloc[:, 1:13].copy()
    x.columns = FEATURE_NAMES
    for col in x.columns:
        x[col] = (
            x[col]
            .astype(str)
            .str.replace(" ", "", regex=False)
            .str.replace(",", "", regex=False)
            .astype(float)
        )
    return regions, x


def save_scatter(scores, regions, explained):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(scores[:, 0], scores[:, 1], s=42, color="#2f6f9f")
    for x, y, name in zip(scores[:, 0], scores[:, 1], regions):
        ax.annotate(name, (x, y), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.2f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.2f}%)")
    ax.set_title("PCA二维得分图")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_scores.png", bbox_inches="tight")
    plt.close(fig)


def save_variance_plot(explained):
    x = np.arange(1, len(explained) + 1)
    cumulative = np.cumsum(explained)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    ax.bar(x, explained * 100, color="#5b8db8", label="方差贡献率")
    ax.plot(x, cumulative * 100, marker="o", color="#c44e52", label="累计贡献率")
    ax.set_xlabel("主成分")
    ax.set_ylabel("贡献率(%)")
    ax.set_title("PCA方差贡献率")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_variance.png", bbox_inches="tight")
    plt.close(fig)


def save_score_bar(result):
    plot_df = result.sort_values("综合得分", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 8), dpi=150)
    ax.barh(plot_df["地区"], plot_df["综合得分"], color="#4c956c")
    ax.set_xlabel("PCA综合得分")
    ax.set_title("各地区公用事业发展水平PCA综合得分")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_comprehensive_score.png", bbox_inches="tight")
    plt.close(fig)


def main():
    setup_plot_style()
    regions, x = load_data()
    z = StandardScaler().fit_transform(x)

    pca = PCA()
    scores = pca.fit_transform(z)
    explained = pca.explained_variance_ratio_

    m = int(np.argmax(np.cumsum(explained) >= 0.85) + 1)
    weights = explained[:m] / explained[:m].sum()
    composite = scores[:, :m] @ weights
    if np.corrcoef(composite, z.sum(axis=1))[0, 1] < 0:
        composite = -composite
        scores[:, 0] = -scores[:, 0]

    result = pd.DataFrame(
        {
            "地区": regions,
            "PC1": scores[:, 0],
            "PC2": scores[:, 1],
            "综合得分": composite,
            "排名": pd.Series(composite).rank(ascending=False, method="min").astype(int),
        }
    ).sort_values("排名")
    result.to_csv("pca_scores.csv", index=False, encoding="utf-8-sig")

    loadings = pd.DataFrame(
        pca.components_.T * np.sqrt(pca.explained_variance_),
        index=FEATURE_NAMES,
        columns=[f"PC{i}" for i in range(1, len(FEATURE_NAMES) + 1)],
    )
    loadings.to_csv("pca_loadings.csv", encoding="utf-8-sig")

    save_scatter(scores[:, :2], regions, explained)
    save_variance_plot(explained)
    save_score_bar(result)

    print("PCA completed.")
    print(f"Selected components for composite score: {m}")
    print("Saved: pca_scores.csv, pca_loadings.csv")


if __name__ == "__main__":
    main()
