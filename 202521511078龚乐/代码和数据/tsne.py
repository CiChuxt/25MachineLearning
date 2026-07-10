from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


DATA_FILE = Path("data.xlsx")
FIG_DIR = Path("figures")
RANDOM_STATE = 42
PERPLEXITY = 5
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
    regions = df.iloc[:, 0].astype(str)
    x = df.iloc[:, 1:13].copy()
    x.columns = FEATURE_NAMES
    for col in x.columns:
        x[col] = x[col].astype(str).str.replace(" ", "", regex=False).str.replace(",", "", regex=False).astype(float)
    return regions, x


def save_embedding(y, regions):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.scatter(y[:, 0], y[:, 1], s=42, color="#ffa600")
    for x0, y0, name in zip(y[:, 0], y[:, 1], regions):
        ax.annotate(name, (x0, y0), xytext=(4, 3), textcoords="offset points", fontsize=8)
    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_title(f"t-SNE二维嵌入图(perplexity={PERPLEXITY})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "tsne_embedding.png", bbox_inches="tight")
    plt.close(fig)


def main():
    setup_plot_style()
    regions, x = load_data()
    z = StandardScaler().fit_transform(x)

    model = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        init="pca",
        learning_rate="auto",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    y = model.fit_transform(z)

    pd.DataFrame({"地区": regions, "TSNE1": y[:, 0], "TSNE2": y[:, 1]}).to_csv(
        "tsne_embedding.csv", index=False, encoding="utf-8-sig"
    )
    save_embedding(y, regions)
    print("t-SNE completed.")
    print(f"perplexity={PERPLEXITY}, random_state={RANDOM_STATE}")


if __name__ == "__main__":
    main()
