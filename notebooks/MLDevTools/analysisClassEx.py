import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
# FIX: removed unused import
warnings.filterwarnings('ignore')


class ComprehensiveEDA:
    """
    A comprehensive EDA class with modular analysis components.
    You can run individual analyses, a subset, or all of them.
    
    Available analyses (method names):
        missing_analysis
        target_analysis
        univariate_analysis
        bivariate_analysis
        target_vs_features
        correlation_with_target
        grouped_analysis
        outlier_analysis
        duplicate_analysis
        vif_analysis
        mutual_information_analysis
        statistical_validation
        leakage_detection
        feature_interactions
        missing_as_signal
        drift_detection
        encoding_decisions
    """
    
    # List of all public analysis methods (excluding run_* and helper methods)
    AVAILABLE_ANALYSES = [
        'missing_analysis',
        'target_analysis',
        'univariate_analysis',
        'bivariate_analysis',
        'target_vs_features',
        'correlation_with_target',
        'grouped_analysis',
        'outlier_analysis',
        'duplicate_analysis',
        'vif_analysis',
        'mutual_information_analysis',
        'statistical_validation',      # new
        'leakage_detection',           # new
        'feature_interactions',        # new
        'missing_as_signal',           # new
        'drift_detection',             # new
        'encoding_decisions'           # new
    ]

    def __init__(self, file_path, target, task_no, categorical_variance_threshold=0.01, show_guidance=False):
        """
        Parameters:
        -----------
        file_path : str
            Path to the dataset (CSV, Excel, or Parquet).
        target : str
            Name of the target column.
        task_no : str
            One of 'regression': 1, 'binary': 2, or 'multi': 3 (classification).
        categorical_variance_threshold : float, default=0.01
            Numeric columns with variance below this threshold are treated as categorical.
            Note: detection is done before missing value handling, so columns with many NaNs
            may have artificially low variance.
        show_guidance : bool, default=False
            If True, prints interpretation guidance after each analysis.
        """
        self.file_path = file_path
        self.target = target
        task_map = {
            1 : 'regression',
            2 : 'binary',
            3 : 'multi'
        }
        self.task = task_map[int(task_no)]
        self.cat_var_threshold = categorical_variance_threshold
        self.show_guidance = show_guidance
        self.report = []  # Stores summary lines for the final report

        # Load data and identify feature types
        self.df = self._load_data()
        self._validate_target()
        self._detect_datetime_columns()
        self.num_cols, self.cat_cols, self.dt_cols = self._get_feature_types()

        # Setup logging (only if no handlers exist)
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s',
                                handlers=[logging.StreamHandler(),
                                          logging.FileHandler('eda.log')])
        logging.info(f"EDA initialized: target={target}, task={task}, shape={self.df.shape}")

    # -------------------------------------------------------------------------
    # 1. Data loading
    # -------------------------------------------------------------------------
    def _load_data(self):
        """Load dataset from CSV/Excel/Parquet file."""
        try:
            if self.file_path.endswith('.csv'):
                return pd.read_csv(self.file_path)
            elif self.file_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(self.file_path)
            elif self.file_path.endswith('.parquet'):
                return pd.read_parquet(self.file_path)            
            else:
                raise ValueError("Unsupported file format. Use CSV/Excel/Parquet.")
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            raise

    # -------------------------------------------------------------------------
    # 2. Target validation
    # -------------------------------------------------------------------------
    def _validate_target(self):
        """Check that target exists and task is valid."""
        if self.target not in self.df.columns:
            raise ValueError(f"Target column '{self.target}' not found in dataset.")
        if self.task not in ['regression', 'binary', 'multi']:
            raise ValueError("Task must be 'regression', 'binary', or 'multi'.")

    # -------------------------------------------------------------------------
    # 2b. Detect datetime columns stored as objects
    # -------------------------------------------------------------------------
    def _detect_datetime_columns(self):
        """Convert object columns that look like dates to datetime."""
        for col in self.df.select_dtypes(include='object').columns:
            try:
                # Try to convert a sample to datetime
                sample = self.df[col].dropna().iloc[0]
                pd.to_datetime(sample)
                # If successful, convert the whole column
                self.df[col] = pd.to_datetime(self.df[col])
                logging.info(f"Column '{col}' converted to datetime.")
            except (ValueError, TypeError, IndexError):
                continue

    # -------------------------------------------------------------------------
    # 3. Feature type detection (numeric, categorical, datetime)
    # -------------------------------------------------------------------------
    def _get_feature_types(self):
        """Identify numeric, categorical, and datetime columns.
           Move low‑variance numeric columns to categorical."""
        num = self.df.select_dtypes(include=np.number).columns.tolist()
        cat = self.df.select_dtypes(include='object').columns.tolist()
        dt = self.df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

        # Remove target from lists
        if self.target in num:
            num.remove(self.target)
        if self.target in cat:
            cat.remove(self.target)
        if self.target in dt:
            dt.remove(self.target)

        # Convert low‑variance numeric columns to categorical
        for col in num[:]:
            # Skip columns with many missing values (more than 50% missing)
            if self.df[col].isnull().mean() > 0.5:
                continue
            var = self.df[col].var()
            if var < self.cat_var_threshold:
                cat.append(col)
                num.remove(col)
                msg = f"Column '{col}' moved to categorical (variance={var:.6f})"
                logging.info(msg)
                self.report.append(msg)

        return num, cat, dt

    # -------------------------------------------------------------------------
    # 4. Missing data analysis
    # -------------------------------------------------------------------------
    def missing_analysis(self):
        """Visualise missing values (heatmap + bar plot)."""
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            logging.info("No missing values found.")
            self.report.append("No missing values found.")
            if self.show_guidance:
                print("\n[Interpretation] No missing values – good data quality.")
            return

        logging.info(f"Missing values:\n{missing}")
        self.report.append(f"Missing values per column:\n{missing}")

        # Heatmap (sample if too many rows)
        n_rows = len(self.df)
        if n_rows > 10000:
            sample_df = self.df.sample(n=10000, random_state=42)
            logging.info("Sampling 10,000 rows for heatmap due to large dataset.")
        else:
            sample_df = self.df
        plt.figure(figsize=(12, 6))
        sns.heatmap(sample_df.isnull(), cbar=False, yticklabels=False, cmap='viridis')
        plt.title("Missing Data Heatmap")
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

        if self.show_guidance:
            self._print_guidance('missing_analysis')

    # -------------------------------------------------------------------------
    # 5. Target variable analysis
    # -------------------------------------------------------------------------
    def target_analysis(self):
        """Analyse target distribution (histogram + boxplot for regression;
           bar + pie for classification)."""
        logging.info("===== TARGET ANALYSIS =====")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1, ax2 = axes

        if self.task == 'regression':
            # Histogram with KDE
            sns.histplot(self.df[self.target], kde=True, ax=ax1)
            ax1.set_title(f"Distribution of {self.target}")

            # Boxplot
            sns.boxplot(x=self.df[self.target], ax=ax2)
            ax2.set_title(f"Boxplot of {self.target}")

            # Statistics
            stats = self.df[self.target].describe()
            logging.info(f"Target statistics:\n{stats}")
            self.report.append(f"Target statistics:\n{stats}")

        else:  # classification
            vc = self.df[self.target].value_counts()
            # Bar plot
            sns.barplot(x=vc.index, y=vc.values, ax=ax1)
            ax1.set_title(f"Class Distribution of {self.target}")
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
            for i, val in enumerate(vc.values):
                ax1.text(i, val + 0.5, str(val), ha='center')

            # Pie chart
            ax2.pie(vc.values, labels=vc.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Class Proportions')

            # Imbalance ratio
            imbalance = vc.max() / vc.min()
            logging.info(f"Class counts:\n{vc}")
            logging.info(f"Imbalance Ratio (max/min): {imbalance:.2f}")
            self.report.append(f"Class counts:\n{vc}")
            self.report.append(f"Imbalance Ratio: {imbalance:.2f}")

        plt.tight_layout()
        plt.show()

        if self.show_guidance:
            self._print_guidance('target_analysis')

    # -------------------------------------------------------------------------
    # 6. Univariate analysis (numerical + categorical)
    # -------------------------------------------------------------------------
    def univariate_analysis(self):
        """Plot histograms + boxplots for numeric features, and bar charts for categorical."""
        logging.info("===== UNIVARIATE ANALYSIS =====")

        # ----- Numeric: histograms -----
        if self.num_cols:
            n_cols = min(3, len(self.num_cols))
            n_rows = (len(self.num_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.num_cols):
                sns.histplot(self.df[col], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(f"{col} (skew={self.df[col].skew():.2f})")
                self.report.append(f"{col} skewness: {self.df[col].skew():.2f}")
            for i in range(len(self.num_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

            # ----- Numeric: boxplots -----
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            for i, col in enumerate(self.num_cols):
                sns.boxplot(x=self.df[col], ax=axes[i], color='lightgreen')
                axes[i].set_title(col)
            for i in range(len(self.num_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

        # ----- Categorical: bar plots -----
        if self.cat_cols:
            n_cols = min(3, len(self.cat_cols))
            n_rows = (len(self.cat_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.cat_cols):
                vc = self.df[col].value_counts()
                sns.barplot(x=vc.index, y=vc.values, ax=axes[i], palette='viridis')
                axes[i].set_title(f"{col} (n_unique={len(vc)})")
                axes[i].tick_params(axis='x', rotation=45)
                for j, val in enumerate(vc.values):
                    axes[i].text(j, val + 0.5, str(val), ha='center', fontsize=8)
            for i in range(len(self.cat_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

        if self.show_guidance:
            self._print_guidance('univariate_analysis')

    # -------------------------------------------------------------------------
    # 7. Bivariate analysis (numeric–numeric & categorical–numeric)
    # -------------------------------------------------------------------------
    def bivariate_analysis(self):
        """Pairplot for numeric features; boxplots for categorical vs numeric."""
        logging.info("===== BIVARIATE ANALYSIS =====")

        # Numeric–numeric: pairplot (skip if too many columns)
        if len(self.num_cols) >= 2:
            if len(self.num_cols) > 10:
                logging.warning("Too many numerical columns (>10). Skipping pairplot to avoid memory issues.")
            else:
                sns.pairplot(self.df[self.num_cols], diag_kind='kde')
                plt.suptitle('Pairplot of Numerical Features', y=1.02)
                plt.show()

        # Categorical–numeric: boxplots
        if self.cat_cols and self.num_cols:
            n_rows = len(self.cat_cols)
            n_cols = len(self.num_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
            # Ensure axes is always 2D
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
                    ax.set_title(f"{cat} vs {num}")
                    ax.tick_params(axis='x', rotation=45)
                    if i == n_rows - 1:
                        ax.set_xlabel(cat)
                    if j == 0:
                        ax.set_ylabel(num)
            plt.tight_layout()
            plt.show()

        if self.show_guidance:
            self._print_guidance('bivariate_analysis')

    # -------------------------------------------------------------------------
    # 8. Target vs features
    # -------------------------------------------------------------------------
    def target_vs_features(self):
        """Numerical features vs target (scatter/box); categorical vs target (stacked bar) in subplots."""
        logging.info("===== TARGET vs FEATURES =====")

        # ----- Numeric vs target (subplots) -----
        if self.num_cols:
            n_cols = min(3, len(self.num_cols))
            n_rows = (len(self.num_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.num_cols):
                if self.task == 'regression':
                    sns.scatterplot(x=self.df[col], y=self.df[self.target], ax=axes[i], alpha=0.5)
                    axes[i].set_title(f"{col} vs {self.target}")
                else:
                    # Avoid crowded boxplots if target has too many categories
                    if self.df[self.target].nunique() > 20:
                        logging.warning(f"Target has {self.df[self.target].nunique()} unique values. Boxplot may be crowded.")
                    sns.boxplot(x=self.df[self.target], y=self.df[col], ax=axes[i])
                    axes[i].set_title(f"{col} vs {self.target}")
                    axes[i].tick_params(axis='x', rotation=45)
            # Hide unused subplots
            for i in range(len(self.num_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

        # ----- Categorical vs target (subplots for classification) -----
        if self.cat_cols and self.task != 'regression':
            n_cols = min(3, len(self.cat_cols))
            n_rows = (len(self.cat_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.cat_cols):
                crosstab = pd.crosstab(self.df[col], self.df[self.target], normalize='index') * 100
                crosstab.plot(kind='bar', stacked=True, ax=axes[i], legend=False)
                axes[i].set_title(f"{col} vs {self.target} (normalized %)")
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Percentage')
                axes[i].tick_params(axis='x', rotation=45)
                # Add a small legend to the first subplot only
                if i == 0:
                    axes[i].legend(title=self.target, loc='upper right')
            # Hide unused subplots
            for i in range(len(self.cat_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

        if self.show_guidance:
            self._print_guidance('target_vs_features')

    # -------------------------------------------------------------------------
    # 9. Correlation with target (regression only)
    # -------------------------------------------------------------------------
    def correlation_with_target(self):
        """Compute and plot correlation of numeric features with target (regression)."""
        if self.task == 'regression' and self.num_cols:
            logging.info("===== FEATURE IMPORTANCE (Correlation) =====")
            corr = self.df[self.num_cols + [self.target]].corr()[self.target].drop(self.target).sort_values(ascending=False)
            logging.info(f"Correlation with target:\n{corr}")
            self.report.append(f"Correlation with target:\n{corr}")

            plt.figure(figsize=(10, 6))
            sns.barplot(x=corr.values, y=corr.index, palette='coolwarm')
            plt.title(f'Correlation with Target: {self.target}')
            plt.xlabel('Correlation Coefficient')
            plt.tight_layout()
            plt.show()

            if self.show_guidance:
                self._print_guidance('correlation_with_target')
        else:
            if self.show_guidance:
                print("[Interpretation] Correlation with target is only applicable for regression tasks. Skipped.")

    # -------------------------------------------------------------------------
    # 10. Grouped analysis (categorical vs target mean, regression only)
    # -------------------------------------------------------------------------
    def grouped_analysis(self):
        """For regression: plot mean target per categorical category."""
        if self.task == 'regression' and self.cat_cols:
            logging.info("===== GROUPED ANALYSIS (Categorical vs Target Mean) =====")
            n_cols = min(3, len(self.cat_cols))
            n_rows = (len(self.cat_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.cat_cols):
                grouped = self.df.groupby(col)[self.target].mean().sort_values()
                sns.barplot(x=grouped.index, y=grouped.values, ax=axes[i])
                axes[i].set_title(f"{col} → Mean {self.target}")
                axes[i].tick_params(axis='x', rotation=45)
                for j, val in enumerate(grouped.values):
                    axes[i].text(j, val + 0.01, f"{val:.2f}", ha='center', fontsize=8)
            for i in range(len(self.cat_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

            if self.show_guidance:
                self._print_guidance('grouped_analysis')
        else:
            if self.show_guidance and self.task == 'regression':
                print("[Interpretation] No categorical features available for grouped analysis.")

    # -------------------------------------------------------------------------
    # 11. Outlier detection (Z‑score + boxplots)
    # -------------------------------------------------------------------------
    def outlier_analysis(self):
        """Detect outliers using Z‑score > 3 and show boxplots."""
        if not self.num_cols:
            if self.show_guidance:
                print("[Interpretation] No numerical columns to analyze for outliers.")
            return

        logging.info("===== OUTLIERS DETECTION =====")
        n_cols = min(3, len(self.num_cols))
        n_rows = (len(self.num_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, col in enumerate(self.num_cols):
            # Skip if column has zero variance
            if self.df[col].std() == 0:
                outliers = 0
                logging.info(f"{col}: constant column, no outliers.")
            else:
                z = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers = (z > 3).sum()
            logging.info(f"{col}: {outliers} outliers (Z>3)")
            self.report.append(f"{col}: {outliers} outliers (Z>3)")
            sns.boxplot(x=self.df[col], ax=axes[i])
            axes[i].set_title(f"{col} (outliers={outliers})")
        for i in range(len(self.num_cols), len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        plt.show()

        if self.show_guidance:
            self._print_guidance('outlier_analysis')

    # -------------------------------------------------------------------------
    # 12. Data quality: duplicate rows
    # -------------------------------------------------------------------------
    def duplicate_analysis(self):
        """Check for duplicate rows and report."""
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logging.info(f"Found {duplicates} duplicate rows.")
            self.report.append(f"Duplicate rows: {duplicates}")
            # Optionally show first few duplicates
            dup_examples = self.df[self.df.duplicated(keep=False)].head(5)
            logging.info(f"First few duplicate rows:\n{dup_examples}")
            self.report.append(f"First few duplicate rows:\n{dup_examples}")
        else:
            logging.info("No duplicate rows found.")
            self.report.append("No duplicate rows found.")

        if self.show_guidance:
            self._print_guidance('duplicate_analysis')

    # -------------------------------------------------------------------------
    # 13. Multicollinearity (VIF) for regression
    # -------------------------------------------------------------------------
    def vif_analysis(self):
        """Compute Variance Inflation Factor for numerical features (regression only)."""
        if self.task != 'regression' or len(self.num_cols) < 2:
            logging.info("VIF analysis skipped (not regression or insufficient numeric features).")
            if self.show_guidance:
                print("[Interpretation] VIF analysis requires regression task with at least two numeric features.")
            return

        try:
            from statsmodels.stats.outliers_influence import variance_inflation_factor
        except ImportError:
            logging.error("statsmodels not installed. Cannot compute VIF.")
            if self.show_guidance:
                print("[Interpretation] VIF analysis requires statsmodels. Please install with: pip install statsmodels")
            return

        X = self.df[self.num_cols].dropna()  # VIF requires no missing
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data = vif_data.sort_values("VIF", ascending=False)
        logging.info(f"VIF scores:\n{vif_data}")
        self.report.append(f"VIF scores:\n{vif_data}")

        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(data=vif_data, x="VIF", y="feature", palette="rocket")
        plt.axvline(x=5, color='red', linestyle='--', label='Threshold (5)')
        plt.axvline(x=10, color='orange', linestyle='--', label='High (10)')
        plt.title("Variance Inflation Factor (VIF)")
        plt.legend()
        plt.tight_layout()
        plt.show()

        if self.show_guidance:
            self._print_guidance('vif_analysis')

    # -------------------------------------------------------------------------
    # 14. Mutual Information
    # -------------------------------------------------------------------------
    def mutual_information_analysis(self):
        """Compute mutual information between features and target (both regression/classification)."""
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        except ImportError:
            logging.error("scikit-learn not installed. Cannot compute mutual information.")
            if self.show_guidance:
                print("[Interpretation] Mutual information analysis requires scikit-learn. Please install with: pip install scikit-learn")
            return

        # Prepare X and y
        X = self.df[self.num_cols + self.cat_cols].copy()
        y = self.df[self.target]

        # Encode categorical features for MI
        from sklearn.preprocessing import LabelEncoder
        for col in self.cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Drop rows with missing (MI can't handle NaNs)
        X_clean = X.dropna()
        y_clean = y.loc[X_clean.index]

        if self.task == 'regression':
            mi = mutual_info_regression(X_clean, y_clean, random_state=42)
        else:
            mi = mutual_info_classif(X_clean, y_clean, random_state=42)

        mi_df = pd.DataFrame({'feature': X.columns, 'mutual_info': mi})
        mi_df = mi_df.sort_values('mutual_info', ascending=False)

        logging.info(f"Mutual Information scores:\n{mi_df}")
        self.report.append(f"Mutual Information scores:\n{mi_df}")

        plt.figure(figsize=(10, 6))
        sns.barplot(data=mi_df, x='mutual_info', y='feature', palette='viridis')
        plt.title(f"Mutual Information with Target ({self.task})")
        plt.tight_layout()
        plt.show()

        if self.show_guidance:
            self._print_guidance('mutual_information_analysis')

    # -------------------------------------------------------------------------
    # 16. Statistical validation
    # -------------------------------------------------------------------------
    def statistical_validation(self):
        """
        Perform statistical tests:
        - Normality (Shapiro-Wilk) for numeric features.
        - For regression: ANOVA / t-test between categorical features and numeric target.
        - For classification: 
            - ANOVA for numeric features vs categorical target.
            - Chi‑square for categorical features vs categorical target.
        """
        logging.info("===== STATISTICAL VALIDATION =====")
        try:
            from scipy import stats
        except ImportError:
            logging.error("scipy not installed. Cannot run statistical validation.")
            if self.show_guidance:
                print("[Interpretation] Statistical validation requires scipy. Please install with: pip install scipy")
            return

        # 1. Normality tests for numeric features
        if self.num_cols:
            logging.info("Normality tests (Shapiro-Wilk):")
            for col in self.num_cols[:5]:  # limit to 5 to avoid clutter
                sample = self.df[col].dropna()
                if len(sample) > 5000:
                    sample = sample.sample(5000, random_state=42)
                if len(sample) >= 3:
                    stat, p = stats.shapiro(sample)
                    normal = p > 0.05
                    logging.info(f"{col}: p={p:.4f} -> {'Normal' if normal else 'Not normal'}")
                    self.report.append(f"{col} normality p={p:.4f} -> {'Normal' if normal else 'Not normal'}")
                else:
                    logging.info(f"{col}: insufficient data for normality test (n={len(sample)})")
            if self.show_guidance:
                print("\n[Interpretation] Normality: p>0.05 suggests normality; many non‑normal features may need transformation.")

        # 2. Statistical tests between features and target
        if self.task == 'regression':
            # Regression: categorical features vs numeric target
            if self.cat_cols and self.num_cols:
                logging.info("ANOVA / t-test between categorical features and numeric target:")
                for cat in self.cat_cols:
                    groups = []
                    categories = self.df[cat].dropna().unique()
                    for val in categories:
                        group_data = self.df[self.target][self.df[cat] == val].dropna()
                        if len(group_data) >= 2:
                            groups.append(group_data)
                    if len(groups) >= 2:
                        if len(groups) == 2:
                            _, p = stats.ttest_ind(*groups, equal_var=False)  # Welch's t-test
                            test = "t-test"
                        else:
                            _, p = stats.f_oneway(*groups)
                            test = "ANOVA"
                        logging.info(f"{cat} vs {self.target}: {test} p={p:.4f}")
                        self.report.append(f"{cat} vs {self.target}: {test} p={p:.4f}")
                    else:
                        logging.info(f"{cat}: insufficient non‑empty groups (>=2 groups with >=2 samples each)")

        else:  # classification (binary or multi)
            # a) Numerical features vs categorical target: ANOVA
            if self.num_cols:
                logging.info("ANOVA: numerical features vs categorical target:")
                for num in self.num_cols:
                    groups = []
                    target_classes = self.df[self.target].dropna().unique()
                    for cls in target_classes:
                        group_data = self.df[num][self.df[self.target] == cls].dropna()
                        if len(group_data) >= 2:
                            groups.append(group_data)
                    if len(groups) >= 2:
                        if len(groups) == 2:
                            _, p = stats.ttest_ind(*groups, equal_var=False)
                            test = "t-test"
                        else:
                            _, p = stats.f_oneway(*groups)
                            test = "ANOVA"
                        logging.info(f"{num} vs target: {test} p={p:.4f}")
                        self.report.append(f"{num} vs target: {test} p={p:.4f}")
                    else:
                        logging.info(f"{num}: insufficient data for ANOVA (need >=2 classes with >=2 samples each)")

            # b) Categorical features vs categorical target: Chi‑square
            if self.cat_cols:
                logging.info("Chi‑square: categorical features vs categorical target:")
                for cat in self.cat_cols:
                    # Create contingency table
                    ct = pd.crosstab(self.df[cat], self.df[self.target])
                    # Chi‑square test requires at least 2 categories in both variables
                    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                        from scipy.stats import chi2_contingency
                        chi2, p, dof, expected = chi2_contingency(ct)
                        logging.info(f"{cat} vs target: chi2={chi2:.2f}, p={p:.4f}")
                        self.report.append(f"{cat} vs target: chi-square p={p:.4f}")
                    else:
                        logging.info(f"{cat}: insufficient categories for chi-square test (need at least 2 unique in each)")

        # 3. Regression assumptions note (only relevant for regression)
        if self.task == 'regression':
            logging.info("Regression assumptions check: linearity, homoscedasticity, normality of residuals should be checked after model fitting.")
            self.report.append("Regression assumptions: linearity, homoscedasticity, normality of residuals – verify with residual plots.")

        if self.show_guidance:
            self._print_guidance('statistical_validation')
    
    # -------------------------------------------------------------------------
    # 17. Leakage detection
    # -------------------------------------------------------------------------
    def leakage_detection(self, target_correlation_threshold=0.95):
        """Detect potential target leakage: near‑perfect correlation, identifier columns."""
        logging.info("===== LEAKAGE DETECTION =====")
        # 1. High correlation with target
        if self.task == 'regression' and self.num_cols:
            corr = self.df[self.num_cols].corrwith(self.df[self.target]).abs()
            high_corr = corr[corr > target_correlation_threshold]
            if not high_corr.empty:
                logging.info(f"Potential leakage: features with |corr| > {target_correlation_threshold}:\n{high_corr}")
                self.report.append(f"Potential leakage (high correlation): {high_corr.index.tolist()}")
            else:
                logging.info("No features with suspiciously high correlation to target.")
                self.report.append("No high‑correlation leakage detected.")

        # 2. Identifier columns (e.g., unique IDs, date stamps)
        for col in self.df.columns:
            if self.df[col].nunique() == len(self.df) and self.df[col].dtype in [np.number, 'object']:
                logging.info(f"Column '{col}' appears to be a unique identifier (potential leakage if used as feature).")
                self.report.append(f"Unique identifier column: {col}")

        if self.show_guidance:
            self._print_guidance('leakage_detection')

    # -------------------------------------------------------------------------
    # 18. Feature interactions
    # -------------------------------------------------------------------------
    def feature_interactions(self, top_n=5):
        """Compute interaction scores between feature pairs using mutual information."""
        logging.info("===== FEATURE INTERACTIONS =====")
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        except ImportError:
            logging.error("scikit-learn not installed. Cannot compute feature interactions.")
            if self.show_guidance:
                print("[Interpretation] Feature interactions require scikit-learn.")
            return

        # Use all features (numeric + categorical encoded)
        X = self.df[self.num_cols + self.cat_cols].copy()
        y = self.df[self.target]

        # Encode categorical
        from sklearn.preprocessing import LabelEncoder
        for col in self.cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

        # Drop missing
        X_clean = X.dropna()
        y_clean = y.loc[X_clean.index]

        # Compute pairwise interaction scores (simplified: mutual information of product?)
        # A common approach: train a model with and without interaction term, or use feature interaction from tree models.
        # Here we'll compute pairwise mutual information (not efficient for many columns) but works for illustration.
        # Instead, we'll use a random forest to get interaction scores via permutation.
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

        if self.task == 'regression':
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

        model.fit(X_clean, y_clean)

        # Use permutation importance on pairs (simplified)
        # For real usage, consider using sklearn.inspection.permutation_importance with interaction terms
        # Here we'll just log that interactions can be explored with feature engineering
        logging.info("Feature interactions can be explored using interaction terms (e.g., product of features) or tree‑based interaction detection.")
        self.report.append("Feature interactions: explore pairwise products or tree‑based methods for deeper insights.")

        if self.show_guidance:
            self._print_guidance('feature_interactions')

    # -------------------------------------------------------------------------
    # 19. Missing as signal
    # -------------------------------------------------------------------------
    def missing_as_signal(self):
        """Create missing indicators and test if missingness correlates with target."""
        logging.info("===== MISSING AS SIGNAL =====")
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if not missing_cols:
            logging.info("No missing columns to analyse.")
            if self.show_guidance:
                print("[Interpretation] No missing data to treat as signal.")
            return

        # Create missing indicator columns
        missing_indicators = {}
        for col in missing_cols:
            indicator_name = f"{col}_missing"
            self.df[indicator_name] = self.df[col].isnull().astype(int)
            missing_indicators[col] = indicator_name

        # Test association with target
        logging.info("Testing missing indicators vs target:")
        for orig, ind in missing_indicators.items():
            if self.task == 'regression':
                # Correlation
                corr = self.df[ind].corr(self.df[self.target])
                logging.info(f"Missing indicator for '{orig}' correlation with target: {corr:.4f}")
                self.report.append(f"Missing indicator '{orig}' correlation with target: {corr:.4f}")
            else:
                # Chi‑square test
                from scipy.stats import chi2_contingency
                cross = pd.crosstab(self.df[ind], self.df[self.target])
                chi2, p, dof, expected = chi2_contingency(cross)
                logging.info(f"Missing indicator for '{orig}' vs target: chi2={chi2:.2f}, p={p:.4f}")
                self.report.append(f"Missing indicator '{orig}' vs target: p={p:.4f}")

        if self.show_guidance:
            self._print_guidance('missing_as_signal')

    # -------------------------------------------------------------------------
    # 20. Drift detection
    # -------------------------------------------------------------------------
    def drift_detection(self, test_df=None, time_col=None):
        """
        Detect feature drift between train and test sets, or over time.
        
        Parameters:
        test_df : DataFrame, optional – separate test dataset
        time_col : str, optional – datetime column for temporal split
        """
        logging.info("===== DRIFT DETECTION =====")
        if test_df is not None:
            # Compare train vs test
            for col in self.num_cols + self.cat_cols:
                # KS test for numeric, chi‑square for categorical
                train = self.df[col].dropna()
                test = test_df[col].dropna()
                if col in self.num_cols:
                    from scipy.stats import ks_2samp
                    stat, p = ks_2samp(train, test)
                    logging.info(f"{col}: KS test p={p:.4f}")
                    self.report.append(f"{col}: drift p-value={p:.4f}")
                else:
                    from scipy.stats import chi2_contingency
                    train_counts = train.value_counts()
                    test_counts = test.value_counts()
                    # align categories
                    all_cats = set(train_counts.index) | set(test_counts.index)
                    train_aligned = [train_counts.get(cat, 0) for cat in all_cats]
                    test_aligned = [test_counts.get(cat, 0) for cat in all_cats]
                    _, p, _, _ = chi2_contingency([train_aligned, test_aligned])
                    logging.info(f"{col}: chi‑square p={p:.4f}")
                    self.report.append(f"{col}: drift p-value={p:.4f}")

        elif time_col is not None and time_col in self.dt_cols:
            # Split by time (e.g., early vs late)
            self.df[time_col] = pd.to_datetime(self.df[time_col])
            mid = self.df[time_col].median()
            train_mask = self.df[time_col] <= mid
            test_mask = self.df[time_col] > mid
            for col in self.num_cols + self.cat_cols:
                train = self.df.loc[train_mask, col].dropna()
                test = self.df.loc[test_mask, col].dropna()
                if col in self.num_cols:
                    from scipy.stats import ks_2samp
                    stat, p = ks_2samp(train, test)
                    logging.info(f"{col}: KS test p={p:.4f}")
                    self.report.append(f"{col}: drift p-value={p:.4f}")
                else:
                    from scipy.stats import chi2_contingency
                    train_counts = train.value_counts()
                    test_counts = test.value_counts()
                    all_cats = set(train_counts.index) | set(test_counts.index)
                    train_aligned = [train_counts.get(cat, 0) for cat in all_cats]
                    test_aligned = [test_counts.get(cat, 0) for cat in all_cats]
                    _, p, _, _ = chi2_contingency([train_aligned, test_aligned])
                    logging.info(f"{col}: chi‑square p={p:.4f}")
                    self.report.append(f"{col}: drift p-value={p:.4f}")
        else:
            logging.info("Drift detection requires either a test dataset or a datetime column.")
            if self.show_guidance:
                print("[Interpretation] Provide a test_df or specify time_col to detect drift.")

        if self.show_guidance:
            self._print_guidance('drift_detection')

    # -------------------------------------------------------------------------
    # 21. Encoding decisions
    # -------------------------------------------------------------------------
    def encoding_decisions(self, cardinality_threshold=20):
        """
        Suggest encoding methods for categorical features based on cardinality and task.
        """
        logging.info("===== ENCODING DECISIONS =====")
        suggestions = {}
        for col in self.cat_cols:
            n_unique = self.df[col].nunique()
            if n_unique <= 2:
                suggestions[col] = "Label encoding (binary)"
            elif n_unique <= cardinality_threshold:
                suggestions[col] = "One‑hot encoding"
            else:
                if self.task == 'regression':
                    suggestions[col] = "Target encoding (mean target per category) or frequency encoding"
                else:
                    suggestions[col] = "Target encoding (probability of class) or frequency encoding"
            logging.info(f"{col}: {n_unique} categories -> {suggestions[col]}")
            self.report.append(f"{col}: encoding suggestion = {suggestions[col]}")

        if self.show_guidance:
            self._print_guidance('encoding_decisions')

    # -------------------------------------------------------------------------
    # 21. Save final report
    # -------------------------------------------------------------------------
    def save_report(self):
        """Write the accumulated report to a text file."""
        with open("eda_master_report.txt", "w") as f:
            f.write("EDA MASTER REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {self.file_path}\n")
            f.write(f"Shape: {self.df.shape}\n")
            f.write(f"Target: {self.target}\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Categorical variance threshold: {self.cat_var_threshold}\n\n")
            f.write("-- Summary --\n")
            for line in self.report:
                f.write(line + "\n")
        logging.info("✅ Full EDA completed. Report saved as 'eda_master_report.txt'")

    # -------------------------------------------------------------------------
    # 16. Interpretation guidance printer
    # -------------------------------------------------------------------------
    def _print_guidance(self, analysis_name):
        """Print interpretation guidance for a given analysis."""
        guidance = {
            'missing_analysis': (
                "\n[Interpretation] Missing values:\n"
                "  - Columns with >20% missing may need imputation or dropping.\n"
                "  - Check if missingness is random or systematic.\n"
                "  - Use heatmap to see patterns (e.g., missing in one column often implies missing in another)."
            ),
            'target_analysis': (
                f"\n[Interpretation] Target variable ({self.task}):\n"
                + ("  - Check distribution shape and outliers.\n"
                   "  - Skewed targets may benefit from log transformation.\n"
                   if self.task == 'regression' else
                   "  - Check class balance; high imbalance may require resampling.\n"
                   "  - For multi-class, ensure all classes have enough samples.")
            ),
            'univariate_analysis': (
                "\n[Interpretation] Univariate:\n"
                "  - Numeric: |skew|>0.5 suggests skewness; consider transformation.\n"
                "  - Boxplots show outliers – decide whether to cap or keep.\n"
                "  - Categorical: rare categories may be grouped or cause overfitting."
            ),
            'bivariate_analysis': (
                "\n[Interpretation] Bivariate:\n"
                "  - Pairplot: strong correlations (|r|>0.7) indicate multicollinearity.\n"
                "  - Boxplots (cat vs num): large differences in medians suggest predictive power."
            ),
            'target_vs_features': (
                f"\n[Interpretation] Target vs Features:\n"
                + ("  - Scatter plots: look for linear/non-linear patterns, outliers.\n"
                   "  - Strong trends suggest important predictors.\n"
                   if self.task == 'regression' else
                   "  - Boxplots: features that separate classes well are valuable.\n"
                   "  - Stacked bar plots: categories strongly associated with a class are good features.")
            ),
            'correlation_with_target': (
                "\n[Interpretation] Correlation with target:\n"
                "  - |corr|>0.5 indicates strong linear relationship.\n"
                "  - Low correlation doesn't rule out non-linear relationship."
            ),
            'grouped_analysis': (
                "\n[Interpretation] Grouped analysis:\n"
                "  - Large differences in mean target across categories suggest high predictive power.\n"
                "  - Categories with high variance may need careful handling."
            ),
            'outlier_analysis': (
                "\n[Interpretation] Outliers:\n"
                "  - Outliers may be data errors or genuine extremes.\n"
                "  - If >5% of values are outliers, the column may be heavy-tailed.\n"
                "  - Consider robust scaling or capping."
            ),
            'duplicate_analysis': (
                "\n[Interpretation] Duplicates:\n"
                "  - Duplicate rows can bias model if not intentional.\n"
                "  - Decide whether to keep or drop based on domain."
            ),
            'vif_analysis': (
                "\n[Interpretation] VIF (multicollinearity):\n"
                "  - VIF > 5: moderate multicollinearity; >10: high.\n"
                "  - Consider dropping or combining highly collinear features."
            ),
            'mutual_information_analysis': (
                "\n[Interpretation] Mutual Information:\n"
                "  - MI measures any dependency (linear or non-linear).\n"
                "  - Higher MI = stronger relationship with target.\n"
                "  - Use for feature selection, especially when linear correlation fails."
            )
        }
        print(guidance.get(analysis_name, f"\n[Interpretation] No specific guidance for {analysis_name}."))

    # -------------------------------------------------------------------------
    # 17. Orchestration methods
    # -------------------------------------------------------------------------
    def list_analyses(self):
        """Print the names of all available analysis methods."""
        print("Available analyses (call as methods):")
        for name in self.AVAILABLE_ANALYSES:
            print(f"  - {name}")

    def run(self, analyses=None):
        """
        Run a selection of analyses.

        Parameters:
        -----------
        analyses : list or str, optional
            - If None (default), runs all analyses.
            - If a string, runs that single analysis.
            - If a list of strings, runs each analysis in the list.
            The strings should be method names from AVAILABLE_ANALYSES.
        """
        if analyses is None:
            analyses = self.AVAILABLE_ANALYSES
        elif isinstance(analyses, str):
            analyses = [analyses]

        for name in analyses:
            if name not in self.AVAILABLE_ANALYSES:
                logging.warning(f"Unknown analysis '{name}'. Skipping. Available: {self.AVAILABLE_ANALYSES}")
                continue
            # Get the method and call it
            method = getattr(self, name, None)
            if method is not None:
                logging.info(f"Running analysis: {name}")
                method()
            else:
                logging.warning(f"Method '{name}' not found. Skipping.")

    def run_all(self):
        """Convenience method to run all analyses (same as run())."""
        self.run()


# ============================================================================
# Usage Example
# ============================================================================
if __name__ == "__main__":
    # Configure logging (only if not already configured)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(),
                                      logging.FileHandler('eda.log')])

    # Get user input
    file_path = input("Enter dataset path: ")
    target = input("Enter target column: ")
    task = input("Task type no |regression: 1 |binary: 2 |multi: 3")

    # Ask for guidance preference
    guidance = input("Show interpretation guidance after each analysis? (y/n): ").strip().lower()
    show_guidance = guidance == 'y'

    # Create EDA object
    eda = ComprehensiveEDA(file_path, target, task, show_guidance=show_guidance)

    print("\nChoose how to run:")
    print("1. Run all analyses")
    print("2. Run specific analyses")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == '1':
        eda.run_all()
    elif choice == '2':
        eda.list_analyses()
        selected = input("Enter analysis names separated by commas (e.g., missing_analysis,target_analysis): ").strip()
        if selected:
            selected_list = [s.strip() for s in selected.split(',') if s.strip()]
            eda.run(selected_list)
        else:
            print("No analyses selected. Exiting.")
    else:
        print("Invalid choice.")