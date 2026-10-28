"""Run the full analysis and save charts + a findings summary.

    python scripts/make_report.py

Outputs:
  reports/figures/*.png
  reports/findings.md   <- copy the key numbers from here into the README
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import analysis, model  # noqa: E402
from src.clean import load_clean  # noqa: E402

FIG = ROOT / "reports" / "figures"
sns.set_theme(style="whitegrid")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=150)
    plt.close(fig)


def main(df=None):
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_clean() if df is None else df
    ov = analysis.overview(df)
    online = analysis.compare_groups(df, "online_order")
    booking = analysis.compare_groups(df, "book_table")
    areas = analysis.by_area(df)
    cuisines = analysis.top_cuisines(df)
    price = analysis.rating_by_price(df)
    gaps = analysis.opportunity_gaps(df)
    m = model.train(df)

    # 1. Rating distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(df["rating"].dropna(), bins=30, ax=ax, color="#2f6fed")
    ax.set(title="Distribution of restaurant ratings", xlabel="Rating", ylabel="Restaurants")
    save(fig, "01_rating_distribution.png")

    # 2. Online ordering and table booking vs rating
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, col, title in [(axes[0], "online_order", "Online ordering"),
                           (axes[1], "book_table", "Table booking")]:
        sns.boxplot(data=df, x=col, y="rating", ax=ax, color="#9db8f5")
        ax.set(title=title, xlabel="", ylabel="Rating")
        ax.set_xticks([0, 1], ["No", "Yes"])
    save(fig, "02_features_vs_rating.png")

    # 3. Areas
    fig, ax = plt.subplots(figsize=(8, 5))
    areas.sort_values("restaurants").plot.barh(y="restaurants", ax=ax, legend=False, color="#2f6fed")
    ax.set(title="Areas with the most restaurants", xlabel="Restaurants", ylabel="")
    save(fig, "03_top_areas.png")

    # 4. Cuisines
    fig, ax = plt.subplots(figsize=(8, 5))
    cuisines.sort_values("restaurants").plot.barh(y="restaurants", ax=ax, legend=False, color="#2f6fed")
    ax.set(title="Most common cuisines", xlabel="Restaurants offering it", ylabel="")
    save(fig, "04_top_cuisines.png")

    # 5. Price band vs rating
    fig, ax = plt.subplots(figsize=(7, 4))
    price["share_rated_4_plus"].plot.bar(ax=ax, color="#2f6fed", rot=0)
    ax.set(title="Share of restaurants rated 4.0+ by cost for two (INR)",
           xlabel="Cost for two", ylabel="Share rated 4.0+")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    save(fig, "05_price_vs_rating.png")

    # 6. Feature importance
    fig, ax = plt.subplots(figsize=(8, 5))
    m["importance"].sort_values().plot.barh(ax=ax, color="#2f6fed")
    ax.set(title="What predicts a 4.0+ rating? (random forest importance)", xlabel="Importance")
    save(fig, "06_feature_importance.png")

    def sig(p):
        return "statistically significant" if p < 0.05 else "not statistically significant"

    lines = [
        "# Findings", "",
        f"- **{ov['restaurants']:,} unique restaurants** across {ov['areas']} areas after removing "
        f"duplicate listings. {ov['rated_share']:.0%} have a rating; median rating "
        f"{ov['median_rating']:.1f}, median cost for two ₹{ov['median_cost_for_two']:.0f}.",
        f"- **Online ordering:** {ov['online_order_share']:.0%} of restaurants offer it. Average rating "
        f"{online['with']:.2f} with vs {online['without']:.2f} without "
        f"({online['difference']:+.2f}, p={online['p_value']:.2g}, {sig(online['p_value'])}).",
        f"- **Table booking:** {ov['book_table_share']:.0%} offer it. Average rating "
        f"{booking['with']:.2f} with vs {booking['without']:.2f} without "
        f"({booking['difference']:+.2f}, p={booking['p_value']:.2g}, {sig(booking['p_value'])}).",
        f"- **Busiest area:** {areas.index[0]} ({areas['restaurants'].iloc[0]} restaurants). "
        f"Highest-rated large area: {areas['avg_rating'].idxmax()} "
        f"(avg {areas['avg_rating'].max():.2f}).",
        f"- **Most common cuisine:** {cuisines.index[0]} ({cuisines['restaurants'].iloc[0]} restaurants). "
        f"Highest-rated common cuisine: {cuisines['avg_rating'].idxmax()} "
        f"(avg {cuisines['avg_rating'].max():.2f}).",
        "- **Price vs rating:** share rated 4.0+ by cost band: " + ", ".join(
            f"{band} {row.share_rated_4_plus:.0%}" for band, row in price.iterrows()) + ".",
        f"- **Prediction:** a random forest predicts a 4.0+ rating with ROC-AUC "
        f"**{m['roc_auc']:.2f}** on a held-out 20% test set ({m['n_test']:,} restaurants; "
        f"{m['baseline_positive_rate']:.0%} are 4.0+). Top features: "
        + ", ".join(m["importance"].index[:3]) + ".",
        "", "## Opportunity gaps (high engagement, few competitors)", "",
        gaps.to_markdown() if hasattr(gaps, "to_markdown") else gaps.to_string(),
        "", "## Model report", "", "```", m["report"], "```",
    ]
    out = ROOT / "reports" / "findings.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved charts to {FIG} and summary to {out}")


if __name__ == "__main__":
    main()
