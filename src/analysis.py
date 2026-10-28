"""Analysis functions used by the notebook, the report script and the dashboard.

Each function takes the cleaned DataFrame and returns a small summary table.
"""
import pandas as pd
from scipy import stats

MIN_RESTAURANTS = 30  # ignore groups too small to compare fairly


def overview(df: pd.DataFrame) -> dict:
    rated = df["rating"].notna()
    return {
        "restaurants": len(df),
        "areas": df["location"].nunique(),
        "rated_share": rated.mean(),
        "median_rating": df.loc[rated, "rating"].median(),
        "median_cost_for_two": df["cost_for_two"].median(),
        "online_order_share": df["online_order"].mean(),
        "book_table_share": df["book_table"].mean(),
    }


def compare_groups(df: pd.DataFrame, flag: str) -> dict:
    """Compare ratings for restaurants with and without a feature (e.g. online
    ordering) and test whether the difference is statistically significant."""
    rated = df.dropna(subset=["rating"])
    yes = rated.loc[rated[flag], "rating"]
    no = rated.loc[~rated[flag], "rating"]
    # Ratings aren't normally distributed, so use Mann-Whitney U rather than a t-test.
    _, p = stats.mannwhitneyu(yes, no, alternative="two-sided")
    return {"with": yes.mean(), "without": no.mean(), "n_with": len(yes),
            "n_without": len(no), "difference": yes.mean() - no.mean(), "p_value": p}


def by_area(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    g = (df.groupby("location")
           .agg(restaurants=("name", "size"), avg_rating=("rating", "mean"),
                median_cost=("cost_for_two", "median"), total_votes=("votes", "sum"))
           .query("restaurants >= @MIN_RESTAURANTS"))
    return g.sort_values("restaurants", ascending=False).head(top).round(2)


def top_cuisines(df: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    exploded = df.assign(cuisine=df["cuisines"].str.split(",")).explode("cuisine")
    exploded["cuisine"] = exploded["cuisine"].str.strip()
    exploded = exploded[exploded["cuisine"] != ""]
    g = (exploded.groupby("cuisine")
           .agg(restaurants=("name", "size"), avg_rating=("rating", "mean"),
                median_cost=("cost_for_two", "median"))
           .query("restaurants >= @MIN_RESTAURANTS"))
    return g.sort_values("restaurants", ascending=False).head(top).round(2)


def rating_by_price(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("price_band", observed=True)
              .agg(restaurants=("name", "size"), avg_rating=("rating", "mean"),
                   share_rated_4_plus=("rating", lambda r: (r >= 4).mean()))
              .round(3))


def by_restaurant_type(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    g = (df.groupby("restaurant_type")
           .agg(restaurants=("name", "size"), avg_rating=("rating", "mean"),
                median_cost=("cost_for_two", "median"))
           .query("restaurants >= @MIN_RESTAURANTS"))
    return g.sort_values("restaurants", ascending=False).head(top).round(2)


def opportunity_gaps(df: pd.DataFrame, top: int = 10) -> pd.DataFrame:
    """Find area + cuisine combinations where customers are engaged (high
    average votes per restaurant) but there are few restaurants competing.

    opportunity = average votes per restaurant / number of restaurants,
    limited to areas with enough restaurants overall to be meaningful.
    """
    big_areas = df["location"].value_counts()
    big_areas = big_areas[big_areas >= 100].index
    sub = df[df["location"].isin(big_areas) & df["primary_cuisine"].notna()]
    g = (sub.groupby(["location", "primary_cuisine"])
            .agg(restaurants=("name", "size"), avg_votes=("votes", "mean"),
                 avg_rating=("rating", "mean"))
            .query("restaurants >= 3"))
    g["opportunity_score"] = g["avg_votes"] / g["restaurants"]
    return g.sort_values("opportunity_score", ascending=False).head(top).round(2)
