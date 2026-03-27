import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
from scipy.stats import skew

warnings.filterwarnings('ignore')

class EDAAnalysis:
    """
    A comprehensive EDA class for regression and classification tasks.
    Handles missing values, outlier detection, univariate/bivariate analysis,
    and generates a summary report.
    """

    def __init__(self, file_path, target, task, categorical_variance_threshold=0.01):
        """
        Initialize the EDA object.

        Parameters:
        file_path (str): Path to the dataset (CSV or Excel).
        target (str): Name of the target column.
        task (str): 'regression', 'binary', or 'multi' (classification).
        categorical_variance_threshold (float): Variance threshold to treat a numeric column as categorical.
        """
        self.file_path = file_path
        self.target = target
        self.task = task
        self.categorical_variance_threshold = categorical_variance_threshold
        self.report = []  # Store summary messages for final report

        # Load data and separate feature types
        self.df = self.load_data()
        self.num_cols, self.cat_cols, self.dt_cols = self.get_feature_types()

        # Validate target existence and task
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataset.")
        if self.task not in ['regression', 'binary', 'multi']:
            raise ValueError("Task must be 'regression', 'binary', or 'multi'.")

        # Setup logging (already configured in main)
        logging.info(f"EDA initialized: target={target}, task={task}, data shape={self.df.shape}")

    def load_data(self):
        """Load dataset based on file extension."""
        if self.file_path.endswith('.csv'):
            df = pd.read_csv(self.file_path)
        elif self.file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(self.file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        return df

    def get_feature_types(self):
        """Identify numerical, categorical, and datetime columns, and move low‑variance numeric columns to categorical."""
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = self.df.select_dtypes(include='object').columns.tolist()
        dt_cols = self.df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

        # Remove target from feature lists
        if self.target in num_cols:
            num_cols.remove(self.target)
        if self.target in cat_cols:
            cat_cols.remove(self.target)
        if self.target in dt_cols:
            dt_cols.remove(self.target)

        # Convert low‑variance numeric columns to categorical
        for col in num_cols[:]:
            var = self.df[col].var()
            if var < self.categorical_variance_threshold:
                cat_cols.append(col)
                num_cols.remove(col)
                logging.info(f"Column {col} moved to categorical due to low variance ({var:.6f})")
                self.report.append(f"Column {col} moved to categorical due to low variance ({var:.6f})")

        return num_cols, cat_cols, dt_cols

    def missing_data_analysis(self):
        """Analyze and visualize missing data."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            logging.info("No missing values found.")
            self.report.append("No missing values found.")
            return
        logging.info("Missing values:\n" + str(missing))
        self.report.append(f"Missing values per column:\n{missing}")

        # Plot missingness
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title('Missing Data Heatmap')
        plt.tight_layout()
        plt.show()

        # Bar plot of missing percentages
        missing_pct = (missing / len(self.df)) * 100
        plt.figure(figsize=(12, 6))
        missing_pct.sort_values().plot(kind='bar')
        plt.ylabel('Percentage Missing')
        plt.title('Missing Data Percentage by Column')
        plt.tight_layout()
        plt.show()

    def target_analysis(self):
        """Analyze the target variable distribution."""
        logging.info("===== TARGET ANALYSIS =====")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1, ax2 = axes

        if self.task == 'regression':
            sns.histplot(self.df[self.target], kde=True, ax=ax1)
            ax1.set_title(f"Distribution of {self.target} (Regression)")
            stats = self.df[self.target].describe()
            logging.info(f"Target statistics:\n{stats}")
            self.report.append(f"Target statistics:\n{stats}")

            # Boxplot for spread
            sns.boxplot(x=self.df[self.target], ax=ax2)
            ax2.set_title(f"Boxplot of {self.target}")

        else:  # classification
            vc = self.df[self.target].value_counts()
            sns.barplot(x=vc.index, y=vc.values, ax=ax1)
            ax1.set_title(f"Class Distribution of {self.target}")
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
            for i, val in enumerate(vc.values):
                ax1.text(i, val + 0.5, str(val), ha='center')

            # Imbalance ratio
            imbalance_ratio = vc.max() / vc.min()
            logging.info(f"Class counts:\n{vc}")
            logging.info(f"Imbalance Ratio (max/min): {imbalance_ratio:.2f}")
            self.report.append(f"Class counts:\n{vc}")
            self.report.append(f"Imbalance Ratio: {imbalance_ratio:.2f}")

            # Pie chart for visual
            ax2.pie(vc.values, labels=vc.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Class Proportions')

        plt.tight_layout()
        plt.show()

    def univariate_analysis(self):
        """Univariate analysis for numerical and categorical features."""
        logging.info("===== UNIVARIATE ANALYSIS =====")

        # Numerical features: histograms and boxplots
        if self.num_cols:
            n_cols = min(3, len(self.num_cols))
            n_rows = (len(self.num_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.num_cols):
                # Histogram with KDE
                sns.histplot(self.df[col], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(f'{col} (skew={self.df[col].skew():.2f})')
                self.report.append(f"{col} skewness: {self.df[col].skew():.2f}")

                # Add boxplot on same axis? Instead, create separate boxplot figure later
            # Hide unused subplots
            for i in range(len(self.num_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

            # Boxplots for numerical features
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            for i, col in enumerate(self.num_cols):
                sns.boxplot(x=self.df[col], ax=axes[i], color='lightgreen')
                axes[i].set_title(f'{col}')
            for i in range(len(self.num_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

        # Categorical features: bar plots
        if self.cat_cols:
            n_cols = min(3, len(self.cat_cols))
            n_rows = (len(self.cat_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.cat_cols):
                vc = self.df[col].value_counts()
                sns.barplot(x=vc.index, y=vc.values, ax=axes[i], palette='viridis')
                axes[i].set_title(f'{col} (n_unique={len(vc)})')
                axes[i].tick_params(axis='x', rotation=45)
                # Add counts on bars
                for j, val in enumerate(vc.values):
                    axes[i].text(j, val + 0.5, str(val), ha='center', fontsize=8)
            for i in range(len(self.cat_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

    def bivariate_analysis(self):
        """Bivariate analysis: numerical vs numerical and categorical vs numerical."""
        logging.info("===== BIVARIATE ANALYSIS =====")

        # Numerical vs Numerical: pairplot
        if len(self.num_cols) >= 2:
            sns.pairplot(self.df[self.num_cols], diag_kind='kde')
            plt.suptitle('Pairplot of Numerical Features', y=1.02)
            plt.show()

        # Categorical vs Numerical: boxplots
        if self.cat_cols and self.num_cols:
            n_rows = len(self.cat_cols)
            n_cols = len(self.num_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            # If only one row/col, make axes iterable
            if n_rows == 1 and n_cols == 1:
                axes = [[axes]]
            elif n_rows == 1:
                axes = [axes]
            elif n_cols == 1:
                axes = [[ax] for ax in axes]

            for i, cat in enumerate(self.cat_cols):
                for j, num in enumerate(self.num_cols):
                    ax = axes[i][j]
                    sns.boxplot(x=self.df[cat], y=self.df[num], ax=ax)
                    ax.set_title(f'{cat} vs {num}')
                    ax.tick_params(axis='x', rotation=45)
                    if i == len(self.cat_cols) - 1:
                        ax.set_xlabel(cat)
                    if j == 0:
                        ax.set_ylabel(num)
            plt.tight_layout()
            plt.show()

    def target_vs_features(self):
        """Relationship between target and features."""
        logging.info("===== TARGET vs FEATURES =====")

        # Numerical features vs target
        if self.num_cols:
            n_cols = min(3, len(self.num_cols))
            n_rows = (len(self.num_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.num_cols):
                if self.task == 'regression':
                    sns.scatterplot(x=self.df[col], y=self.df[self.target], ax=axes[i], alpha=0.5)
                    axes[i].set_title(f'{col} vs {self.target}')
                else:
                    # Classification: boxplot by target class
                    sns.boxplot(x=self.df[self.target], y=self.df[col], ax=axes[i])
                    axes[i].set_title(f'{col} vs {self.target}')
                    axes[i].tick_params(axis='x', rotation=45)
            for i in range(len(self.num_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

        # Categorical features vs target (classification) – use grouped bar plots
        if self.cat_cols and self.task != 'regression':
            for cat in self.cat_cols:
                crosstab = pd.crosstab(self.df[cat], self.df[self.target], normalize='index') * 100
                crosstab.plot(kind='bar', stacked=True, figsize=(10, 6))
                plt.title(f'{cat} vs {self.target} (normalized %)')
                plt.xlabel(cat)
                plt.ylabel('Percentage')
                plt.legend(title=self.target)
                plt.tight_layout()
                plt.show()

    def correlation_with_target(self):
        """Compute correlation with target (only for regression)."""
        if self.task == 'regression' and self.num_cols:
            logging.info("===== FEATURE IMPORTANCE (Correlation) =====")
            corr = self.df[self.num_cols + [self.target]].corr()[self.target].drop(self.target).sort_values(ascending=False)
            logging.info(f"Correlation with target:\n{corr}")
            self.report.append(f"Correlation with target:\n{corr}")

            # Plot
            plt.figure(figsize=(10, 6))
            sns.barplot(x=corr.values, y=corr.index, palette='coolwarm')
            plt.title(f'Correlation with Target: {self.target}')
            plt.xlabel('Correlation Coefficient')
            plt.tight_layout()
            plt.show()

    def grouped_analysis(self):
        """Group by categorical features and compute target mean."""
        if self.cat_cols and self.task == 'regression':
            logging.info("===== GROUPED ANALYSIS (Categorical vs Target Mean) =====")
            n_cols = min(3, len(self.cat_cols))
            n_rows = (len(self.cat_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.cat_cols):
                grouped = self.df.groupby(col)[self.target].mean().sort_values()
                sns.barplot(x=grouped.index, y=grouped.values, ax=axes[i])
                axes[i].set_title(f'{col} → Mean {self.target}')
                axes[i].tick_params(axis='x', rotation=45)
                # Add value labels
                for j, val in enumerate(grouped.values):
                    axes[i].text(j, val + 0.01, f'{val:.2f}', ha='center', fontsize=8)
            for i in range(len(self.cat_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

    def outliers_detection(self):
        """Detect outliers using Z‑score and visualize boxplots."""
        logging.info("===== OUTLIERS DETECTION =====")
        if not self.num_cols:
            return

        n_cols = min(3, len(self.num_cols))
        n_rows = (len(self.num_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, col in enumerate(self.num_cols):
            z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            outliers = (z_scores > 3).sum()
            logging.info(f"{col}: {outliers} outliers (Z>3)")
            self.report.append(f"{col}: {outliers} outliers (Z>3)")
            sns.boxplot(x=self.df[col], ax=axes[i])
            axes[i].set_title(f"{col} (outliers={outliers})")
        for i in range(len(self.num_cols), len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.show()

    def save_report(self):
        """Save the report to a text file."""
        with open("eda_master_report.txt", "w") as f:
            f.write("EDA MASTER REPORT\n")
            f.write("="*50 + "\n\n")
            f.write(f"Dataset: {self.file_path}\n")
            f.write(f"Shape: {self.df.shape}\n")
            f.write(f"Target: {self.target}\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Categorical variance threshold: {self.categorical_variance_threshold}\n\n")
            f.write("-- Summary --\n")
            for line in self.report:
                f.write(line + "\n")
        logging.info("✅ FULL EDA COMPLETED. Report saved as 'eda_master_report.txt'")

    def execute(self):
        """Run all analysis steps."""
        self.missing_data_analysis()
        self.target_analysis()
        self.univariate_analysis()
        self.bivariate_analysis()
        self.target_vs_features()
        self.correlation_with_target()
        self.grouped_analysis()
        self.outliers_detection()
        self.save_report()


# ============================================================================
# Usage Example
# ============================================================================
if __name__ == "__main__":
    # Configure logging to both console and file
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler('eda.log')])

    # User input
    file_path = input("Enter dataset path: ")
    target = input("Enter target column: ")
    task = input("Task type (regression/binary/multi): ")

    try:
        eda = EDAAnalysis(file_path, target, task)
        eda.execute()
    except Exception as e:
        logging.error(f"Error: {e}")
        print(f"Error: {e}")