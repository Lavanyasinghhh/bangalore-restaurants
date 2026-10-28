"""Generates a small FAKE dataset in the same raw format as zomato.csv.

Used only so the tests can run without downloading the real data. None of
the numbers in the README come from this.
"""
import numpy as np
import pandas as pd


def make_fake_raw(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    locations = ["Koramangala", "BTM", "HSR", "Indiranagar", "Whitefield"]
    types = ["Quick Bites", "Casual Dining", "Cafe", "Delivery"]
    cuisines = ["North Indian", "Chinese", "South Indian", "Cafe", "Biryani", "Pizza"]
    rows = []
    for i in range(n):
        online = rng.random() < 0.6
        book = rng.random() < 0.15
        cost = int(rng.choice([200, 300, 400, 600, 800, 1200, 2500]))
        r = rng.random()
        rate = "NEW" if r < 0.05 else ("-" if r < 0.08 else
               f"{np.clip(rng.normal(3.7 + 0.3 * book, 0.4), 1.8, 4.9):.1f}/5")
        rows.append({
            "url": f"http://example.com/{i}", "address": f"{i} Fake St",
            "name": f"Restaurant {i}", "online_order": "Yes" if online else "No",
            "book_table": "Yes" if book else "No", "rate": rate,
            "votes": int(rng.integers(0, 2000)), "phone": "000",
            "location": rng.choice(locations), "rest_type": rng.choice(types),
            "dish_liked": "", "cuisines": ", ".join(rng.choice(cuisines, size=rng.integers(1, 4), replace=False)),
            "approx_cost(for two people)": f"{cost:,}", "reviews_list": "[]", "menu_item": "[]",
            "listed_in(type)": "Delivery", "listed_in(city)": "Koramangala",
        })
    df = pd.DataFrame(rows)
    # Duplicate some rows under another listing type, like the real data.
    dup = df.sample(200, random_state=seed).assign(**{"listed_in(type)": "Dine-out"})
    return pd.concat([df, dup], ignore_index=True)
