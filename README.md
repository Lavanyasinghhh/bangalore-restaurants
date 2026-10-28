# What makes a restaurant succeed in Bangalore?

[![tests](https://github.com/Lavanyasinghhh/bangalore-restaurants/actions/workflows/tests.yml/badge.svg)](https://github.com/Lavanyasinghhh/bangalore-restaurants/actions/workflows/tests.yml)

An end-to-end analysis of around 12,000 restaurants listed on Zomato in Bangalore. It covers data cleaning, exploratory analysis, statistical testing, a rating-prediction model, and an interactive Streamlit dashboard for exploring the market.

This extends a university project where I analysed the same data and presented it in Tableau. Here I've restructured the code into reusable modules, added statistical tests and a prediction model, and built the dashboard in Python.

## Business questions

1. Does offering **online ordering** or **table booking** go with better ratings?
2. Which **areas and cuisines** dominate the market, and which are rated highest?
3. Do more **expensive** restaurants get better ratings?
4. Can we **predict** whether a restaurant will be rated 4.0+ from its basic attributes?
5. Where are the **gaps**: area and cuisine combinations with engaged customers but few competitors?

## Approach

**Cleaning** (`src/clean.py`)

- Ratings are stored as text like `"4.1/5"`, with `"NEW"` or `"-"` for unrated places. I convert these to numbers and treat `"NEW"` as missing rather than zero.
- Costs are stored as text with commas (`"1,200"`), so I convert them to numbers.
- Each restaurant appears once per listing category (Delivery, Dine-out and so on). I removed these duplicates by name and address. Otherwise, restaurants listed in more categories would be over-counted.

**Analysis** (`src/analysis.py`)

- I compare groups with a **Mann-Whitney U test**, because ratings are skewed and not normally distributed.
- Areas and cuisines with fewer than 30 restaurants are excluded from rankings, so small groups don't top the charts by chance.
- **Opportunity score** = average votes per restaurant ÷ number of restaurants, for each area and primary cuisine. A high score means customers are engaged but there are few competitors.

**Model** (`src/model.py`)

- I trained a random forest classifier to predict a rating of 4.0 or higher, evaluated with ROC-AUC on a stratified 20% held-out test set.
- **Votes are deliberately excluded.** They build up alongside the rating, so a new restaurant wouldn't have them, and including them would leak information about the target.

**Dashboard** (`dashboard/app.py`)

- Built with Streamlit, with filters for area, restaurant type, price and online ordering.
- It shows KPIs, cuisine and price charts, an area table, opportunity gaps, and the top-rated restaurants.

## Run it

```bash
pip install -r requirements.txt
# download zomato.csv from Kaggle into data/ (see data/README.md)

python scripts/make_report.py         # charts → reports/figures, summary → reports/findings.md
jupyter notebook notebooks/analysis.ipynb
streamlit run dashboard/app.py
python -m unittest                    # tests run on a small generated sample, no download needed
```

## Project structure

```
src/
  clean.py        # loading + cleaning
  analysis.py     # summary tables and statistical tests
  model.py        # rating prediction model
notebooks/
  analysis.ipynb  # the full analysis with commentary
dashboard/
  app.py          # Streamlit dashboard
scripts/
  make_report.py  # runs everything, saves charts and findings
tests/            # unit tests (use generated fake data, not the real dataset)
```

## Limitations

- The data is a 2019 snapshot from a single platform.
- Ratings are self-selected: people who leave ratings aren't a random sample of diners.
- The results show correlation, not causation. For example, restaurants that take bookings are often pricier sit-down venues, which may explain some of their higher ratings.

**Tech:** Python, pandas, NumPy, SciPy, scikit-learn, Matplotlib, Seaborn, Streamlit

**Data:** [Zomato Bangalore Restaurants](https://www.kaggle.com/datasets/himanshupoddar/zomato-bangalore-restaurants) (Kaggle)
