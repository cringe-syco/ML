# =========================================
# MODULAR EDA FRAMEWORK
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# =========================
# DATA LOADER
# =========================
class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        if self.file_path.endswith('.csv'):
            return pd.read_csv(self.file_path)
        elif self.file_path.endswith(('.xlsx', '.xls')):
            return pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format")

# =========================
# FEATURE TYPE DETECTOR
# =========================
class FeatureAnalyzer:
    def __init__(self, df, target):
        self.df = df
        self.target = target

    def get_types(self):
        num = self.df.select_dtypes(include=np.number).columns.tolist()
        cat = self.df.select_dtypes(include='object').columns.tolist()
        dt = self.df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

        for col in [num, cat, dt]:
            if self.target in col:
                col.remove(self.target)

        return num, cat, dt

# =========================
# MISSING ANALYSIS
# =========================
class MissingAnalyzer:
    def run(self, df, report):
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            report.append("No missing values")
            return

        report.append(f"Missing:\n{missing}")

        sns.heatmap(df.isnull(), cbar=False)
        plt.title("Missing Heatmap")
        plt.show()

# =========================
# TARGET ANALYSIS
# =========================
class TargetAnalyzer:
    def run(self, df, target, task, report):
        if task == 'regression':
            sns.histplot(df[target], kde=True)
            plt.title("Target Distribution")
            plt.show()
            report.append(str(df[target].describe()))
        else:
            vc = df[target].value_counts()
            sns.barplot(x=vc.index, y=vc.values)
            plt.title("Target Distribution")
            plt.show()
            report.append(str(vc))

# =========================
# UNIVARIATE
# =========================
class UnivariateAnalyzer:
    def run(self, df, num_cols, cat_cols, report):
        for col in num_cols:
            sns.histplot(df[col], kde=True)
            plt.title(col)
            plt.show()

        for col in cat_cols:
            df[col].value_counts().plot(kind='bar')
            plt.title(col)
            plt.show()

# =========================
# BIVARIATE
# =========================
class BivariateAnalyzer:
    def run(self, df, num_cols, cat_cols):
        if len(num_cols) > 1:
            sns.pairplot(df[num_cols])
            plt.show()

        for cat in cat_cols:
            for num in num_cols:
                sns.boxplot(x=df[cat], y=df[num])
                plt.show()

# =========================
# TARGET RELATION
# =========================
class TargetRelationAnalyzer:
    def run(self, df, num_cols, cat_cols, target, task):
        for col in num_cols:
            if task == 'regression':
                sns.scatterplot(x=df[col], y=df[target])
            else:
                sns.boxplot(x=df[target], y=df[col])
            plt.show()

        for col in cat_cols:
            pd.crosstab(df[col], df[target]).plot(kind='bar', stacked=True)
            plt.show()

# =========================
# OUTLIER ANALYSIS
# =========================
class OutlierAnalyzer:
    def run(self, df, num_cols, report):
        for col in num_cols:
            z = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers = (z > 3).sum()
            report.append(f"{col}: {outliers} outliers")

# =========================
# REPORT
# =========================
class ReportManager:
    def save(self, report):
        with open("eda_report.txt", "w") as f:
            for line in report:
                f.write(line + "\n")

# =========================
# ORCHESTRATOR
# =========================
class EDAOrchestrator:
    def __init__(self, file_path, target, task):
        self.df = DataLoader(file_path).load()
        self.target = target
        self.task = task
        self.report = []

        self.num_cols, self.cat_cols, self.dt_cols = FeatureAnalyzer(self.df, target).get_types()

    def run_all(self):
        MissingAnalyzer().run(self.df, self.report)
        TargetAnalyzer().run(self.df, self.target, self.task, self.report)
        UnivariateAnalyzer().run(self.df, self.num_cols, self.cat_cols, self.report)
        BivariateAnalyzer().run(self.df, self.num_cols, self.cat_cols)
        TargetRelationAnalyzer().run(self.df, self.num_cols, self.cat_cols, self.target, self.task)
        OutlierAnalyzer().run(self.df, self.num_cols, self.report)
        ReportManager().save(self.report)

# =========================
# USAGE
# =========================
if __name__ == "__main__":
    file_path = input("Dataset path: ")
    target = input("Target column: ")
    task = input("Task type: ")

    eda = EDAOrchestrator(file_path, target, task)

    # Option 1: Run everything
    eda.run_all()

    # Option 2: Run specific parts
    # MissingAnalyzer().run(eda.df, eda.report)
    # UnivariateAnalyzer().run(eda.df, eda.num_cols, eda.cat_cols, eda.report)