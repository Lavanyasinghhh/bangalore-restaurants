"""Load and clean the Zomato Bangalore restaurants dataset.

Raw data: https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants
(download zomato.csv into data/).

Main cleaning steps:
- ratings like "4.1/5" -> 4.1; "NEW" and "-" -> missing
- costs like "1,200" -> 1200
- Yes/No columns -> booleans
- the same restaurant appears once per listing category (Delivery, Dine-out,
  ...), so duplicates are removed by name + address
- drop heavy text columns (reviews, menus) that aren't used in the analysis
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "zomato.csv"

RENAME = {
    "approx_cost(for two people)": "cost_for_two",
    "listed_in(type)": "listing_type",
    "listed_in(city)": "area",
    "rest_type": "restaurant_type",
}
DROP = ["url", "phone", "reviews_list", "menu_item", "dish_liked"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{path} not found. Download zomato.csv from Kaggle "
            "(himanshupoddar/zomato-bangalore-restaurants) into the data/ folder."
        )
    return pd.read_csv(path, usecols=lambda c: c not in ("reviews_list", "menu_item"))


def parse_rating(value) -> float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().replace("/5", "").strip()
    try:
        return float(text)
    except ValueError:  # "NEW", "-"
        return np.nan


def parse_cost(value) -> float:
    if pd.isna(value):
        return np.nan
    try:
        return float(str(value).replace(",", "").strip())
    except ValueError:
        return np.nan


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.drop(columns=[c for c in DROP if c in raw.columns]).rename(columns=RENAME)

    df["rating"] = df["rate"].map(parse_rating)
    df["is_new"] = df["rate"].astype(str).str.strip().str.upper().eq("NEW")
    df = df.drop(columns="rate")
    df["cost_for_two"] = df["cost_for_two"].map(parse_cost)
    for col in ("online_order", "book_table"):
        df[col] = df[col].astype(str).str.strip().str.lower().eq("yes")

    df["name"] = df["name"].astype(str).str.strip()
    df["cuisines"] = df["cuisines"].fillna("").astype(str)
    df["primary_cuisine"] = df["cuisines"].str.split(",").str[0].str.strip().replace("", np.nan)
    df["n_cuisines"] = df["cuisines"].map(lambda s: len([c for c in s.split(",") if c.strip()]))

    # One row per restaurant. Keep the listing with the most votes.
    df = (df.sort_values("votes", ascending=False)
            .drop_duplicates(subset=["name", "address"])
            .reset_index(drop=True))

    df["price_band"] = pd.cut(
        df["cost_for_two"], bins=[0, 300, 600, 1000, 2000, np.inf],
        labels=["Under 300", "300-600", "600-1000", "1000-2000", "2000+"],
    )
    return df


def load_clean(path: Path = RAW_PATH) -> pd.DataFrame:
    return clean(load_raw(path))
