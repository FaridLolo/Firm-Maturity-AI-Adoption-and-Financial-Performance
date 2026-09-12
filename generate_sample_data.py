"""
Generates a FULLY SYNTHETIC sample dataset with the same schema as the
real (ORBIS-licensed) panel data, so the public demo app has something
to show without ever exposing licensed data.

All company names, financials and ratios below are fabricated. Any
resemblance to real Finnish companies is coincidental. Re-run this
script any time to regenerate `sample_data.csv`.
"""

import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# Fictional company names (deliberately generic / not real listed firms)
companies = [
    ("Nordic Retail Group Oyj", 34, 8.9),
    ("Helsinki Digital Services Oy", 22, 7.8),
    ("Baltic Air Transport Oyj", 95, 9.2),
    ("Aurora Interactive Oy", 27, 8.2),
    ("Suomi Media House Oy", 20, 7.7),
    ("Nordica Home & Living Oyj", 31, 7.7),
    ("Fenno Industrial Group Abp", 340, 9.0),
    ("Lakeland Wellness Oyj", 65, 8.4),
    ("Kaira Timber Homes Oy", 58, 7.6),
    ("Vaasa Press Oyj", 110, 9.1),
    ("Metro Auto Trading Oyj", 16, 9.1),
    ("Keski Regional Media Oyj", 145, 8.8),
]

years = [2021, 2022, 2023]
rows = []


def _normalize_weights(p_high):
    # Higher p_high -> more weight on AI scores 2/3, mimicking younger firms
    w = np.array([1 - p_high, 0.6, 0.3 + p_high, 0.2 + p_high])
    return w / w.sum()

for unit, (name, base_age, base_size) in enumerate(companies, start=1):
    # Older firms adopt AI later / less intensively (mirrors the research story)
    ai_base_prob = np.clip(1.6 - base_age / 250, 0.05, 0.95)

    for i, year in enumerate(years):
        age = base_age + i
        size = base_size + rng.normal(0, 0.03)

        ai_score = rng.choice([0, 1, 2, 3], p=_normalize_weights(ai_base_prob))
        maturity = rng.normal(15 - base_age / 40, 8) + i * 0.5
        noise = rng.normal(0, 1)

        roa = 0.6 * maturity / 3 - 0.9 * ai_score + rng.normal(0, 5)
        roe = 2.2 * maturity / 3 - 1.5 * ai_score + rng.normal(0, 12)
        risk = 25 + 18 * ai_score + 4 * maturity + rng.normal(0, 15)
        risk = max(risk, 1)

        total_assets = rng.integers(50_000, 5_000_000)
        retained_earnings = int(total_assets * (maturity / 100))

        rows.append(
            {
                "Company": name,
                "Unit": unit,
                "Year": year,
                "Age": age,
                "Size": round(size, 4),
                "AI": ai_score,
                "ROA": round(roa, 2),
                "ROE": round(roe, 2),
                "RE": retained_earnings,
                "TA": total_assets,
                "Maturity": round(maturity, 2),
                "Risk": round(risk, 2),
            }
        )

df = pd.DataFrame(rows)
df.to_csv("sample_data.csv", index=False)
print("sample_data.csv written:", df.shape)
print(df.head())
