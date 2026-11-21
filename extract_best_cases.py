"""
Extract best cases from CSV files made from the test_other_params.py script.
"""

from pathlib import Path
import pandas as pd

data_path = Path("data")

files = list(data_path.glob("*.csv"))

for file in files:
    df = pd.read_csv(file)
    df.sort_values(by="J_best", inplace=True, ascending=False)
    df = df[df['feasible'] == 1.0]
    print("-"*40)
    print(file)
    print(df.iloc[0])
    print("-"*40)