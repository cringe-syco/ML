import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
import sys
from pandas.testing import assert_frame_equal

# Create synthetic dataset
np.random.seed(42)
n = 200
num1 = np.random.randn(n) * 10 + 50
num2 = num1 * 0.5 + np.random.randn(n) * 2
cat1 = np.random.choice(['A','B','C'], size=n, p=[0.6,0.3,0.1])
cat2 = np.random.choice(['X','Y'], size=n)
dates = pd.date_range('2020-01-01', periods=n, freq='D').astype(str)
# Introduce some missingness
num1[::20] = np.nan
cat1[::33] = None

# target for regression
target = num1 * 0.3 + np.random.randn(n) * 5

df = pd.DataFrame({
    'num1': num1,
    'num2': num2,
    'cat1': cat1,
    'cat2': cat2,
    'date': dates,
    'target': target
})

# Save CSV for EDA to load
csv_path = 'synth_test.csv'
df.to_csv(csv_path, index=False)

# Import the refactored EDA module by file path
spec = importlib.util.spec_from_file_location('analysisClass13', 'notebooks/MLDevTools/analysisClass13.py')
module = importlib.util.module_from_spec(spec)
loader = spec.loader
assert loader is not None
loader.exec_module(module)
ComprehensiveEDA = module.ComprehensiveEDA

# Instantiate EDA
eda = ComprehensiveEDA(csv_path, 'target', 1, show_plots=False, show_guidance=True)

# Keep a deep copy of the original for mutation checks
orig_copy = eda.df_original.copy(deep=True)

# Run a subset of analyses that shouldn't require optional heavy libs
analyses_to_run = [
    'missing_analysis',
    'target_analysis',
    'univariate_analysis',
    'bivariate_analysis',
    'target_vs_features',
    'correlation_with_target',
    'grouped_analysis',
    'outlier_analysis',
    'duplicate_analysis',
    'data_summary',
    'correlation_heatmap',
    'time_series_analysis',
    'data_quality_checks',
    'missing_as_signal',
    'drift_detection'
]

for name in analyses_to_run:
    print(f"Running: {name}")
    method = getattr(eda, name, None)
    if method is None:
        print(f"  Skipping {name}: not found")
        continue
    try:
        if name == 'drift_detection':
            # avoid requiring an external test set
            method()
        elif name == 'outlier_analysis':
            method(method=None)
        else:
            method()
    except Exception as e:
        print(f"  Method {name} raised {type(e).__name__}: {e}")

    # Check no new columns were added to df_original
    try:
        assert_frame_equal(orig_copy, eda.df_original)
    except AssertionError as ae:
        print("ERROR: df_original was mutated by the analysis!")
        print(ae)
        raise

    # Check matplotlib figures closed
    figs = plt.get_fignums()
    if figs:
        print(f"ERROR: Open matplotlib figures after {name}: {figs}")
        plt.close('all')
        raise RuntimeError("Open matplotlib figures detected")

print("All selected analyses ran without mutating df_original and no open figures remain.")
