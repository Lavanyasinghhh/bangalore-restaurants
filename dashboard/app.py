"""Interactive dashboard.  Run from the project root:

    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import analysis  # noqa: E402
from src.clean import load_clean  # noqa: E402

st.set_page_config(page_title="Bangalore Restaurants", layout="wide")


@st.cache_data
def get_data():
    return load_clean()


df = get_data()

# Sidebar filters
st.sidebar.header("Filters")
areas = st.sidebar.multiselect("Area", sorted(df["location"].dropna().unique()))
types = st.sidebar.multiselect("Restaurant type", sorted(df["restaurant_type"].dropna().unique()))
max_cost = int(df["cost_for_two"].max())
cost = st.sidebar.slider("Cost for two (INR)", 0, max_cost, (0, max_cost), step=100)
online_only = st.sidebar.checkbox("Online ordering only")

f = df[df["cost_for_two"].between(*cost) | df["cost_for_two"].isna()]
if areas:
    f = f[f["location"].isin(areas)]
if types:
    f = f[f["restaurant_type"].isin(types)]
if online_only:
    f = f[f["online_order"]]

st.title("Bangalore restaurant market")
st.caption("Zomato listings · Explore how area, cuisine, price and services relate to ratings")

if f.empty:
    st.warning("No restaurants match these filters.")
    st.stop()

ov = analysis.overview(f)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Restaurants", f"{ov['restaurants']:,}")
c2.metric("Median rating", f"{ov['median_rating']:.1f}")
c3.metric("Median cost for two", f"₹{ov['median_cost_for_two']:.0f}")
c4.metric("Offer online ordering", f"{ov['online_order_share']:.0%}")

left, right = st.columns(2)
with left:
    st.subheader("Top cuisines")
    st.bar_chart(analysis.top_cuisines(f, top=10)["restaurants"], horizontal=True)
with right:
    st.subheader("Share rated 4.0+ by price band")
    st.bar_chart(analysis.rating_by_price(f)["share_rated_4_plus"])

left, right = st.columns(2)
with left:
    st.subheader("Average rating: online ordering")
    st.bar_chart(f.groupby("online_order")["rating"].mean().rename({True: "Yes", False: "No"}))
with right:
    st.subheader("Average rating: table booking")
    st.bar_chart(f.groupby("book_table")["rating"].mean().rename({True: "Yes", False: "No"}))

st.subheader("Areas")
st.dataframe(analysis.by_area(f, top=30))

st.subheader("Opportunity gaps")
st.caption("Area + cuisine combinations with high average votes but few competing restaurants")
st.dataframe(analysis.opportunity_gaps(df if not areas else f))

st.subheader("Top rated restaurants (min. 100 votes)")
top = (f[f["votes"] >= 100].sort_values(["rating", "votes"], ascending=False)
       [["name", "location", "restaurant_type", "cuisines", "rating", "votes", "cost_for_two"]]
       .head(20))
st.dataframe(top, hide_index=True)
