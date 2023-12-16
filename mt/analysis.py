import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mt.definitions import DATA_DIR

if __name__ == "__main__":
    df = pd.read_csv(DATA_DIR / "final_non_scaled.csv", index_col=0)
    print(df.columns)
    print(df.describe().to_csv("desc_stat.csv"))
    print(df.isnull().sum())
    df = pd.read_csv(DATA_DIR / "final.csv", index_col=0)
    # Distribution of Y values
    sns.histplot(df["Y"], kde=True)
    plt.title("Distribution of Y")
    plt.savefig("hist.png")
    # # Correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f")
    plt.tight_layout()
    plt.title("Correlation Matrix")
    plt.savefig("corr.png")
    selected_features = [
        "code",
        "comments",
        "complexity",
        "dependencies",
        "pylint_rating",
        "pylint_message_count",
        "comment_tokens",
    ]
    # # Pair plot for a subset of features (select features relevant to your analysis)
    sns.pairplot(df[["Y", "Y_scale", "Y_standard_scaled"] + selected_features])
    plt.savefig("pair.png")
    # Box plot for checking outliers in Y
    # sns.boxplot(data=df, y="Y")
    # plt.title("Box Plot of Y for Outliers")
    # plt.savefig("boxplot.png")
