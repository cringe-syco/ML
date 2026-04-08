import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import r2_score, accuracy_score
import io

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
        categorical_interactions
        feature_combination_predictiveness
        feature_stability_analysis
        single_feature_performance
        monotonicity_analysis
        rare_category_analysis
        redundancy_pruning
        temporal_leakage_check
        feature_recommendation_engine
    """
    
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
        'statistical_validation',
        'leakage_detection',
        'feature_interactions',
        'missing_as_signal',
        'drift_detection',
        'encoding_decisions',
        'categorical_interactions',
        'feature_combination_predictiveness',
        'feature_stability_analysis',
        'single_feature_performance',
        'monotonicity_analysis',
        'rare_category_analysis',
        'redundancy_pruning',
        'temporal_leakage_check',
        'feature_recommendation_engine'
    ]

    def __init__(self, file_path, target, task_no, 
                categorical_variance_threshold=0.01,
                categorical_cardinality_threshold=None,
                show_guidance=False):
        """
        Parameters:
        -----------
        file_path : str
            Path to the dataset (CSV, Excel, or Parquet).
        target : str
            Name of the target column.
        task_no : str or int
            One of 'regression': 1, 'binary classification': 2, or 'multi classification': 3.
        categorical_variance_threshold : float, default=0.01
            Numeric columns with variance below this threshold are treated as categorical.
        categorical_cardinality_threshold : int or None, default=None
            Numeric columns with unique values <= this threshold are treated as categorical.
            If None, the threshold is automatically computed as:
                clamp( int(0.05 * n_rows), min=10, max=50 )
            where n_rows is the number of rows in the dataset.
        show_guidance : bool, default=False
            If True, prints interpretation guidance after each analysis.
        """
        self.file_path = file_path
        self.target = target
        task_map = {
            1: 'regression',
            2: 'binary',
            3: 'multi'
        }
        self.task = task_map[int(task_no)]
        self.cat_var_threshold = categorical_variance_threshold
        self.show_guidance = show_guidance
        self.report = []

        # Load data and identify feature types
        self.df = self._load_data()
        self._validate_target()
        self._detect_datetime_columns()

        # Determine cardinality threshold (auto if None)
        n_rows = len(self.df)
        if categorical_cardinality_threshold is None:
            auto_thresh = int(0.05 * n_rows)
            self.cat_card_threshold = max(10, min(50, auto_thresh))
            logging.info(f"Auto-set categorical cardinality threshold to {self.cat_card_threshold} (5% of {n_rows} rows, clamped to [10,50])")
        else:
            self.cat_card_threshold = categorical_cardinality_threshold
            logging.info(f"Using user-provided categorical cardinality threshold: {self.cat_card_threshold}")

        self.num_cols, self.cat_cols, self.dt_cols = self._get_feature_types()

        # Setup logging (only if no handlers exist)
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s',
                                handlers=[logging.StreamHandler(),
                                        logging.FileHandler('eda.log')])
        logging.info(f"EDA initialized: target={target}, task={self.task}, shape={self.df.shape}")

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
                sample = self.df[col].dropna().iloc[0]
                pd.to_datetime(sample)
                self.df[col] = pd.to_datetime(self.df[col])
                logging.info(f"Column '{col}' converted to datetime.")
            except (ValueError, TypeError, IndexError):
                continue

    # -------------------------------------------------------------------------
    # 3. Feature type detection
    # -------------------------------------------------------------------------
    def _get_feature_types(self):
        """Identify numeric, categorical, and datetime columns.
        Move low-variance or low-cardinality numeric columns to categorical."""
        num = self.df.select_dtypes(include=np.number).columns.tolist()
        cat = self.df.select_dtypes(include='object').columns.tolist()
        dt = self.df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

        if self.target in num:
            num.remove(self.target)
        if self.target in cat:
            cat.remove(self.target)
        if self.target in dt:
            dt.remove(self.target)

        for col in num[:]:
            if self.df[col].isnull().mean() > 0.5:
                continue
            n_unique = self.df[col].nunique()
            if n_unique <= self.cat_card_threshold:
                cat.append(col)
                num.remove(col)
                msg = f"Column '{col}' moved to categorical (unique values={n_unique} <= threshold {self.cat_card_threshold})"
                logging.info(msg)
                self.report.append(msg)
                continue
            var = self.df[col].var()
            if var < self.cat_var_threshold:
                cat.append(col)
                num.remove(col)
                msg = f"Column '{col}' moved to categorical (variance={var:.6f} < threshold {self.cat_var_threshold})"
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
            sns.histplot(self.df[self.target], kde=True, ax=ax1)
            ax1.set_title(f"Distribution of {self.target}")
            sns.boxplot(x=self.df[self.target], ax=ax2)
            ax2.set_title(f"Boxplot of {self.target}")
            stats = self.df[self.target].describe()
            logging.info(f"Target statistics:\n{stats}")
            self.report.append(f"Target statistics:\n{stats}")
        else:
            vc = self.df[self.target].value_counts()
            sns.barplot(x=vc.index, y=vc.values, ax=ax1)
            ax1.set_title(f"Class Distribution of {self.target}")
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
            for i, val in enumerate(vc.values):
                ax1.text(i, val + 0.5, str(val), ha='center')
            ax2.pie(vc.values, labels=vc.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Class Proportions')
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
    # 6. Univariate analysis
    # -------------------------------------------------------------------------
    def univariate_analysis(self):
        """Plot histograms + boxplots for numeric, bar charts for categorical."""
        logging.info("===== UNIVARIATE ANALYSIS =====")

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

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            for i, col in enumerate(self.num_cols):
                sns.boxplot(x=self.df[col], ax=axes[i], color='lightgreen')
                axes[i].set_title(col)
            for i in range(len(self.num_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

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
    # 7. Bivariate analysis
    # -------------------------------------------------------------------------
    def bivariate_analysis(self):
        """Pairplot for numeric; boxplots for categorical vs numeric."""
        logging.info("===== BIVARIATE ANALYSIS =====")

        if len(self.num_cols) >= 2:
            if len(self.num_cols) > 10:
                logging.warning("Too many numerical columns (>10). Skipping pairplot.")
            else:
                sns.pairplot(self.df[self.num_cols], diag_kind='kde')
                plt.suptitle('Pairplot of Numerical Features', y=1.02)
                plt.show()

        if self.cat_cols and self.num_cols:
            n_rows = len(self.cat_cols)
            n_cols = len(self.num_cols)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
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
        """Numerical vs target (scatter/box); categorical vs target (stacked bar)."""
        logging.info("===== TARGET vs FEATURES =====")

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
                    if self.df[self.target].nunique() > 20:
                        logging.warning(f"Target has {self.df[self.target].nunique()} unique values. Boxplot may be crowded.")
                    sns.boxplot(x=self.df[self.target], y=self.df[col], ax=axes[i])
                    axes[i].set_title(f"{col} vs {self.target}")
                    axes[i].tick_params(axis='x', rotation=45)
            for i in range(len(self.num_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            plt.show()

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
                if i == 0:
                    axes[i].legend(title=self.target, loc='upper right')
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
        if self.task == 'regression' and self.cat_cols:
            logging.info("===== GROUPED ANALYSIS (Categorical vs Target Mean) =====")
            n_cols = min(3, len(self.cat_cols))
            n_rows = (len(self.cat_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            for i, col in enumerate(self.cat_cols):
                grouped = self.df.groupby(col)[self.target].mean().sort_values()
                sns.barplot(x=grouped.index, y=grouped.values, ax=axes[i])
                axes[i].set_title(f"{col} -> Mean {self.target}")
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
    # 11. Outlier detection (Z-score + boxplots)
    # -------------------------------------------------------------------------
    def outlier_analysis(self):
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
    # 12. Duplicate analysis
    # -------------------------------------------------------------------------
    def duplicate_analysis(self):
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            logging.info(f"Found {duplicates} duplicate rows.")
            self.report.append(f"Duplicate rows: {duplicates}")
            dup_examples = self.df[self.df.duplicated(keep=False)].head(5)
            logging.info(f"First few duplicate rows:\n{dup_examples}")
            self.report.append(f"First few duplicate rows:\n{dup_examples}")
        else:
            logging.info("No duplicate rows found.")
            self.report.append("No duplicate rows found.")
        if self.show_guidance:
            self._print_guidance('duplicate_analysis')

    # -------------------------------------------------------------------------
    # 13. VIF analysis (regression only)
    # -------------------------------------------------------------------------
    def vif_analysis(self):
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
        X = self.df[self.num_cols].dropna()
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data = vif_data.sort_values("VIF", ascending=False)
        logging.info(f"VIF scores:\n{vif_data}")
        self.report.append(f"VIF scores:\n{vif_data}")
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
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        except ImportError:
            logging.error("scikit-learn not installed. Cannot compute mutual information.")
            if self.show_guidance:
                print("[Interpretation] Mutual information analysis requires scikit-learn. Please install with: pip install scikit-learn")
            return
        X = self.df[self.num_cols + self.cat_cols].copy()
        y = self.df[self.target]
        from sklearn.preprocessing import LabelEncoder
        for col in self.cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
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
    # 15. Statistical validation
    # -------------------------------------------------------------------------
    def statistical_validation(self):
        logging.info("===== STATISTICAL VALIDATION =====")
        try:
            from scipy import stats
        except ImportError:
            logging.error("scipy not installed. Cannot run statistical validation.")
            if self.show_guidance:
                print("[Interpretation] Statistical validation requires scipy. Please install with: pip install scipy")
            return
        if self.num_cols:
            logging.info("Normality tests (Shapiro-Wilk):")
            for col in self.num_cols[:5]:
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
                print("\n[Interpretation] Normality: p>0.05 suggests normality; many non-normal features may need transformation.")
        if self.task == 'regression':
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
                            _, p = stats.ttest_ind(*groups, equal_var=False)
                            test = "t-test"
                        else:
                            _, p = stats.f_oneway(*groups)
                            test = "ANOVA"
                        logging.info(f"{cat} vs {self.target}: {test} p={p:.4f}")
                        self.report.append(f"{cat} vs {self.target}: {test} p={p:.4f}")
                    else:
                        logging.info(f"{cat}: insufficient non-empty groups (>=2 groups with >=2 samples each)")
        else:
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
            if self.cat_cols:
                logging.info("Chi-square: categorical features vs categorical target:")
                for cat in self.cat_cols:
                    ct = pd.crosstab(self.df[cat], self.df[self.target])
                    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
                        chi2, p, dof, expected = stats.chi2_contingency(ct)
                        logging.info(f"{cat} vs target: chi2={chi2:.2f}, p={p:.4f}")
                        self.report.append(f"{cat} vs target: chi-square p={p:.4f}")
                    else:
                        logging.info(f"{cat}: insufficient categories for chi-square test (need at least 2 unique in each)")
        if self.task == 'regression':
            logging.info("Regression assumptions check: linearity, homoscedasticity, normality of residuals should be checked after model fitting.")
            self.report.append("Regression assumptions: linearity, homoscedasticity, normality of residuals – verify with residual plots.")
        if self.show_guidance:
            self._print_guidance('statistical_validation')

    # -------------------------------------------------------------------------
    # 16. Leakage detection
    # -------------------------------------------------------------------------
    def leakage_detection(self, target_correlation_threshold=0.95):
        logging.info("===== LEAKAGE DETECTION =====")
        if self.task == 'regression' and self.num_cols:
            corr = self.df[self.num_cols].corrwith(self.df[self.target]).abs()
            high_corr = corr[corr > target_correlation_threshold]
            if not high_corr.empty:
                logging.info(f"Potential leakage: features with |corr| > {target_correlation_threshold}:\n{high_corr}")
                self.report.append(f"Potential leakage (high correlation): {high_corr.index.tolist()}")
            else:
                logging.info("No features with suspiciously high correlation to target.")
                self.report.append("No high-correlation leakage detected.")
        for col in self.df.columns:
            if self.df[col].nunique() == len(self.df) and self.df[col].dtype in [np.number, 'object']:
                logging.info(f"Column '{col}' appears to be a unique identifier (potential leakage if used as feature).")
                self.report.append(f"Unique identifier column: {col}")
        if self.show_guidance:
            self._print_guidance('leakage_detection')

    # -------------------------------------------------------------------------
    # 17. Feature interactions
    # -------------------------------------------------------------------------
    def feature_interactions(self, top_n=5):
        logging.info("===== FEATURE INTERACTIONS =====")
        try:
            from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
        except ImportError:
            logging.error("scikit-learn not installed. Cannot compute feature interactions.")
            if self.show_guidance:
                print("[Interpretation] Feature interactions require scikit-learn.")
            return
        X = self.df[self.num_cols + self.cat_cols].copy()
        y = self.df[self.target]
        from sklearn.preprocessing import LabelEncoder
        for col in self.cat_cols:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X_clean = X.dropna()
        y_clean = y.loc[X_clean.index]
        from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
        if self.task == 'regression':
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        model.fit(X_clean, y_clean)
        logging.info("Feature interactions can be explored using interaction terms (e.g., product of features) or tree-based interaction detection.")
        self.report.append("Feature interactions: explore pairwise products or tree-based methods for deeper insights.")
        if self.show_guidance:
            self._print_guidance('feature_interactions')

    # -------------------------------------------------------------------------
    # 18. Missing as signal
    # -------------------------------------------------------------------------
    def missing_as_signal(self):
        logging.info("===== MISSING AS SIGNAL =====")
        missing_cols = self.df.columns[self.df.isnull().any()].tolist()
        if not missing_cols:
            logging.info("No missing columns to analyse.")
            if self.show_guidance:
                print("[Interpretation] No missing data to treat as signal.")
            return
        missing_indicators = {}
        for col in missing_cols:
            indicator_name = f"{col}_missing"
            self.df[indicator_name] = self.df[col].isnull().astype(int)
            missing_indicators[col] = indicator_name
        logging.info("Testing missing indicators vs target:")
        for orig, ind in missing_indicators.items():
            if self.task == 'regression':
                corr = self.df[ind].corr(self.df[self.target])
                logging.info(f"Missing indicator for '{orig}' correlation with target: {corr:.4f}")
                self.report.append(f"Missing indicator '{orig}' correlation with target: {corr:.4f}")
            else:
                from scipy.stats import chi2_contingency
                cross = pd.crosstab(self.df[ind], self.df[self.target])
                chi2, p, dof, expected = chi2_contingency(cross)
                logging.info(f"Missing indicator for '{orig}' vs target: chi2={chi2:.2f}, p={p:.4f}")
                self.report.append(f"Missing indicator '{orig}' vs target: p={p:.4f}")
        if self.show_guidance:
            self._print_guidance('missing_as_signal')

    # -------------------------------------------------------------------------
    # 19. Drift detection
    # -------------------------------------------------------------------------
    def drift_detection(self, test_df=None, time_col=None):
        logging.info("===== DRIFT DETECTION =====")
        if test_df is not None:
            for col in self.num_cols + self.cat_cols:
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
                    all_cats = set(train_counts.index) | set(test_counts.index)
                    train_arr = [train_counts.get(cat, 0) for cat in all_cats]
                    test_arr = [test_counts.get(cat, 0) for cat in all_cats]
                    if sum(train_arr) > 0 and sum(test_arr) > 0:
                        _, p, _, _ = chi2_contingency([train_arr, test_arr])
                        logging.info(f"{col}: chi-square p={p:.4f}")
                        self.report.append(f"{col}: drift p-value={p:.4f}")
                    else:
                        logging.info(f"{col}: cannot compute chi2 due to empty categories.")
        elif time_col is not None and time_col in self.dt_cols:
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
                    train_arr = [train_counts.get(cat, 0) for cat in all_cats]
                    test_arr = [test_counts.get(cat, 0) for cat in all_cats]
                    _, p, _, _ = chi2_contingency([train_arr, test_arr])
                    logging.info(f"{col}: chi-square p={p:.4f}")
                    self.report.append(f"{col}: drift p-value={p:.4f}")
        else:
            logging.info("Drift detection requires either a test dataset or a datetime column.")
            if self.show_guidance:
                print("[Interpretation] Provide a test_df or specify time_col to detect drift.")
        if self.show_guidance:
            self._print_guidance('drift_detection')

    # -------------------------------------------------------------------------
    # 20. Encoding decisions
    # -------------------------------------------------------------------------
    def encoding_decisions(self, cardinality_threshold=20):
        logging.info("===== ENCODING DECISIONS =====")
        suggestions = {}
        for col in self.cat_cols:
            n_unique = self.df[col].nunique()
            if n_unique <= 2:
                suggestions[col] = "Label encoding (binary)"
            elif n_unique <= cardinality_threshold:
                suggestions[col] = "One-hot encoding"
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
    # 21. Categorical interactions (classification only)
    # -------------------------------------------------------------------------
    def categorical_interactions(self, top_n=10, max_categories_per_feature=20):
        if self.task == 'regression':
            logging.info("Categorical interactions analysis is intended for classification tasks. Skipping.")
            if self.show_guidance:
                print("[Interpretation] Categorical interactions are most useful for classification. Skipped.")
            return
        if len(self.cat_cols) < 2:
            logging.info("Need at least two categorical columns for interaction analysis.")
            if self.show_guidance:
                print("[Interpretation] Need at least two categorical features for interaction analysis.")
            return
        logging.info("===== CATEGORICAL FEATURE INTERACTIONS =====")
        usable_cats = [col for col in self.cat_cols if self.df[col].nunique() <= max_categories_per_feature]
        if len(usable_cats) < 2:
            logging.info(f"After filtering by cardinality (≤{max_categories_per_feature}), insufficient features.")
            return
        from sklearn.feature_selection import mutual_info_classif
        from sklearn.preprocessing import LabelEncoder
        y = self.df[self.target]
        scores = []
        for i in range(len(usable_cats)):
            for j in range(i+1, len(usable_cats)):
                col1, col2 = usable_cats[i], usable_cats[j]
                combined = self.df[col1].astype(str) + '_' + self.df[col2].astype(str)
                le = LabelEncoder()
                combined_encoded = le.fit_transform(combined)
                mi = mutual_info_classif(combined_encoded.reshape(-1,1), y, random_state=42)[0]
                scores.append((col1, col2, mi))
        if not scores:
            logging.info("No interaction scores computed.")
            return
        scores.sort(key=lambda x: x[2], reverse=True)
        top_pairs = scores[:top_n]
        plt.figure(figsize=(10, 6))
        pair_labels = [f"{p[0]} & {p[1]}" for p in top_pairs]
        mi_vals = [p[2] for p in top_pairs]
        sns.barplot(x=mi_vals, y=pair_labels, palette='Blues_d')
        plt.title(f"Top {top_n} Categorical Feature Pairs by Mutual Information with Target")
        plt.xlabel('Mutual Information')
        plt.tight_layout()
        plt.show()
        best_pair = top_pairs[0]
        col1, col2, mi = best_pair
        logging.info(f"Top interaction: {col1} & {col2} (MI={mi:.4f})")
        self.report.append(f"Top categorical interaction: {col1} & {col2} (MI={mi:.4f})")
        combo = self.df[col1].astype(str) + ' | ' + self.df[col2].astype(str)
        target_dist = pd.crosstab(combo, self.df[self.target], normalize='index')
        max_prop = target_dist.max(axis=1)
        index_df = target_dist.index.to_series().str.split(' | ', expand=True)
        index_df.columns = [col1, col2]
        heatmap_df = pd.DataFrame({
            col1: index_df[col1],
            col2: index_df[col2],
            'max_prop': max_prop.values
        })
        pivot = heatmap_df.pivot(index=col1, columns=col2, values='max_prop')
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5)
        plt.title(f"Proportion of Majority Class in Combinations of\n{col1} (rows) and {col2} (columns)")
        plt.tight_layout()
        plt.show()
        if self.show_guidance:
            self._print_guidance('categorical_interactions')

    # -------------------------------------------------------------------------
    # 22. Feature combination predictiveness (classification only)
    # -------------------------------------------------------------------------
    def feature_combination_predictiveness(self, max_combinations=50, test_size=0.2):
        if self.task == 'regression':
            logging.info("Feature combination analysis is primarily for classification tasks. Skipping.")
            if self.show_guidance:
                print("[Interpretation] Feature combination analysis is most useful for classification. Skipped.")
            return
        if len(self.cat_cols) < 2:
            logging.info("Need at least two categorical columns for combination analysis.")
            return
        logging.info("===== FEATURE COMBINATION PREDICTIVENESS =====")
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.feature_extraction import FeatureHasher
        from sklearn.preprocessing import LabelEncoder
        X_base = self.df[self.num_cols + self.cat_cols].copy()
        for col in self.cat_cols:
            X_base[col] = LabelEncoder().fit_transform(X_base[col].astype(str))
        X_base = X_base.fillna(X_base.median())
        y = self.df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=test_size, random_state=42, stratify=y)
        base_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        base_model.fit(X_train, y_train)
        base_acc = accuracy_score(y_test, base_model.predict(X_test))
        hasher = FeatureHasher(n_features=100, input_type='string')
        all_cats_combo = self.df[self.cat_cols].astype(str).agg('_'.join, axis=1)
        hashed_all = hasher.transform(all_cats_combo.values.reshape(-1,1)).toarray()
        X_all = np.hstack([X_base, hashed_all])
        X_train2, X_test2, y_train2, y_test2 = train_test_split(X_all, y, test_size=test_size, random_state=42, stratify=y)
        combo_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        combo_model.fit(X_train2, y_train2)
        combo_acc = accuracy_score(y_test2, combo_model.predict(X_test2))
        improvement = combo_acc - base_acc
        logging.info(f"Base model accuracy: {base_acc:.4f}")
        logging.info(f"Model with all categorical combinations: {combo_acc:.4f}")
        logging.info(f"Improvement: {improvement:.4f}")
        self.report.append(f"Base model accuracy: {base_acc:.4f}")
        self.report.append(f"Accuracy with categorical combinations: {combo_acc:.4f}")
        self.report.append(f"Improvement: {improvement:.4f}")
        if improvement > 0:
            print(f"\n✅ Adding categorical combinations improved accuracy by {improvement:.4f}.")
        else:
            print(f"\n⚠️ Adding categorical combinations did not improve accuracy (change = {improvement:.4f}).")
        if self.show_guidance:
            self._print_guidance('feature_combination_predictiveness')

    # -------------------------------------------------------------------------
    # 23. Feature stability analysis
    # -------------------------------------------------------------------------
    def feature_stability_analysis(self, test_df=None, bootstrap_n=0):
        logging.info("===== FEATURE STABILITY ANALYSIS =====")
        if test_df is not None:
            from scipy.stats import ks_2samp, chi2_contingency
            for col in self.num_cols + self.cat_cols:
                train = self.df[col].dropna()
                test = test_df[col].dropna()
                if col in self.num_cols:
                    stat, p = ks_2samp(train, test)
                    logging.info(f"{col}: KS test p={p:.4f}")
                    self.report.append(f"Stability {col}: KS p={p:.4f}")
                else:
                    train_counts = train.value_counts()
                    test_counts = test.value_counts()
                    all_cats = set(train_counts.index) | set(test_counts.index)
                    train_arr = [train_counts.get(cat, 0) for cat in all_cats]
                    test_arr = [test_counts.get(cat, 0) for cat in all_cats]
                    if sum(train_arr) > 0 and sum(test_arr) > 0:
                        _, p, _, _ = chi2_contingency([train_arr, test_arr])
                        logging.info(f"{col}: chi2 p={p:.4f}")
                        self.report.append(f"Stability {col}: chi2 p={p:.4f}")
                    else:
                        logging.info(f"{col}: cannot compute chi2 due to empty categories.")
        elif bootstrap_n > 0:
            from sklearn.utils import resample
            n = len(self.df)
            means = {col: [] for col in self.num_cols}
            for _ in range(bootstrap_n):
                boot = resample(self.df, n_samples=n, random_state=None)
                for col in self.num_cols:
                    means[col].append(boot[col].mean())
            for col in self.num_cols:
                var = np.var(means[col])
                logging.info(f"{col}: bootstrap variance of mean = {var:.6f}")
                self.report.append(f"Stability {col}: bootstrap mean variance = {var:.6f}")
        else:
            logging.info("No test set or bootstrap samples provided. Skipping stability analysis.")
            if self.show_guidance:
                print("[Interpretation] Provide a test_df or set bootstrap_n>0 for stability analysis.")

    # -------------------------------------------------------------------------
    # 24. Single feature performance
    # -------------------------------------------------------------------------
    def single_feature_performance(self, test_size=0.2, max_features=None):
        logging.info("===== SINGLE FEATURE PERFORMANCE =====")
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
        from sklearn.metrics import r2_score, accuracy_score
        from sklearn.preprocessing import LabelEncoder
        features_to_test = self.num_cols + self.cat_cols
        if max_features is not None:
            features_to_test = features_to_test[:max_features]
        results = []
        y = self.df[self.target]
        for col in features_to_test:
            X = self.df[[col]].copy()
            if col in self.cat_cols:
                X[col] = LabelEncoder().fit_transform(X[col].astype(str))
            X = X.dropna()
            y_clean = y.loc[X.index]
            if len(X) < 10:
                logging.info(f"{col}: insufficient samples, skipping.")
                continue
            X_train, X_test, y_train, y_test = train_test_split(X, y_clean, test_size=test_size, random_state=42)
            if self.task == 'regression':
                model = DecisionTreeRegressor(max_depth=3, random_state=42)
                model.fit(X_train, y_train)
                score = r2_score(y_test, model.predict(X_test))
            else:
                model = DecisionTreeClassifier(max_depth=3, random_state=42)
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
            results.append((col, score))
        results.sort(key=lambda x: x[1], reverse=True)
        logging.info("Single feature performance (top 10):")
        for col, score in results[:10]:
            logging.info(f"{col}: {score:.4f}")
            self.report.append(f"Single feature {col}: {score:.4f}")
        if self.show_guidance:
            print("\n[Interpretation] Higher scores indicate better predictive power in isolation.")

    # -------------------------------------------------------------------------
    # 25. Monotonicity analysis
    # -------------------------------------------------------------------------
    def monotonicity_analysis(self):
        logging.info("===== MONOTONICITY ANALYSIS =====")
        from scipy.stats import spearmanr
        from sklearn.preprocessing import LabelEncoder
        monotonic = {}
        for col in self.num_cols:
            X = self.df[col].dropna()
            y = self.df.loc[X.index, self.target]
            if self.task == 'regression':
                corr, p = spearmanr(X, y)
                monotonic[col] = {'corr': corr, 'p': p}
                logging.info(f"{col}: Spearman r={corr:.4f}, p={p:.4f}")
                self.report.append(f"Monotonicity {col}: Spearman r={corr:.4f}")
            else:
                le = LabelEncoder()
                y_num = le.fit_transform(y)
                corr, p = spearmanr(X, y_num)
                monotonic[col] = {'corr': corr, 'p': p}
                logging.info(f"{col}: Spearman r (with numeric target) = {corr:.4f}, p={p:.4f}")
                self.report.append(f"Monotonicity {col}: Spearman r={corr:.4f}")
        if self.show_guidance:
            print("\n[Interpretation] |Spearman r| > 0.5 indicates strong monotonic relationship.")

    # -------------------------------------------------------------------------
    # 26. Rare category analysis
    # -------------------------------------------------------------------------
    def rare_category_analysis(self, min_freq=0.01):
        logging.info("===== RARE CATEGORY ANALYSIS =====")
        for col in self.cat_cols:
            vc = self.df[col].value_counts(normalize=True)
            rare = vc[vc < min_freq]
            if not rare.empty:
                logging.info(f"{col}: {len(rare)} rare categories (< {min_freq*100:.1f}%): {rare.index.tolist()}")
                self.report.append(f"Rare categories in {col}: {len(rare)}")
                rare_mask = self.df[col].isin(rare.index)
                if self.task == 'regression':
                    mean_rare = self.df.loc[rare_mask, self.target].mean()
                    mean_other = self.df.loc[~rare_mask, self.target].mean()
                    logging.info(f"  Target mean for rare: {mean_rare:.4f}, other: {mean_other:.4f}")
                    self.report.append(f"  Target mean diff: {mean_rare - mean_other:.4f}")
                else:
                    dist_rare = self.df.loc[rare_mask, self.target].value_counts(normalize=True)
                    dist_other = self.df.loc[~rare_mask, self.target].value_counts(normalize=True)
                    logging.info(f"  Class distribution for rare:\n{dist_rare}")
                    self.report.append(f"  Class distribution for rare: {dist_rare.to_dict()}")
        if self.show_guidance:
            print("\n[Interpretation] Rare categories may cause overfitting; consider grouping or removing them.")

    # -------------------------------------------------------------------------
    # 27. Redundancy pruning
    # -------------------------------------------------------------------------
    def redundancy_pruning(self, corr_threshold=0.85):
        logging.info("===== REDUNDANCY PRUNING =====")
        if len(self.num_cols) < 2:
            logging.info("Need at least two numeric features.")
            return []
        corr_matrix = self.df[self.num_cols].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = []
        if self.task == 'regression':
            target_corr = self.df[self.num_cols].corrwith(self.df[self.target]).abs()
        else:
            target_corr = None
        for col in self.num_cols:
            high_corr = upper[col][upper[col] > corr_threshold].index.tolist()
            for corr_col in high_corr:
                if col not in to_drop and corr_col not in to_drop:
                    if target_corr is not None:
                        if target_corr[col] < target_corr[corr_col]:
                            to_drop.append(col)
                        else:
                            to_drop.append(corr_col)
                    else:
                        to_drop.append(corr_col)
        to_drop = list(set(to_drop))
        logging.info(f"Suggested redundant features to drop: {to_drop}")
        self.report.append(f"Redundant features (|corr|>{corr_threshold}): {to_drop}")
        return to_drop

    # -------------------------------------------------------------------------
    # 28. Temporal leakage check
    # -------------------------------------------------------------------------
    def temporal_leakage_check(self, time_col=None):
        logging.info("===== TEMPORAL LEAKAGE CHECK =====")
        if time_col is None or time_col not in self.dt_cols:
            logging.info("No time column provided. Skipping temporal leakage check.")
            return
        time_num = pd.to_datetime(self.df[time_col]).astype(int) / 10**9
        leakage_candidates = []
        for col in self.num_cols:
            if np.abs(time_num.corr(self.df[col])) > 0.95:
                leakage_candidates.append(col)
                logging.info(f"{col} is highly correlated with time ({time_col}). Potential leakage.")
                self.report.append(f"Potential temporal leakage: {col} correlates with {time_col}.")
        if not leakage_candidates:
            logging.info("No features found to be highly correlated with time.")
        if self.show_guidance:
            print("[Interpretation] Features that are highly correlated with time may cause temporal leakage if used in models.")

    # -------------------------------------------------------------------------
    # 29. Feature recommendation engine
    # -------------------------------------------------------------------------
    def feature_recommendation_engine(self, test_df=None, time_col=None):
        logging.info("===== FEATURE RECOMMENDATION ENGINE =====")
        self.single_feature_performance(max_features=20)
        self.monotonicity_analysis()
        drop_suggestions = self.redundancy_pruning()
        self.rare_category_analysis()
        self.temporal_leakage_check(time_col)
        print("\nFeature Recommendation Summary:")
        print("-" * 50)
        if drop_suggestions:
            print(f"Consider dropping: {drop_suggestions} (redundant)")
        self.report.append("Feature recommendation engine completed.")
        if self.show_guidance:
            self._print_guidance('feature_recommendation_engine')

    # -------------------------------------------------------------------------
    # 30. Save final report
    # -------------------------------------------------------------------------
    def save_report(self):
        with open("eda_master_report.txt", "w", encoding='utf-8') as f:
            f.write("EDA MASTER REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {self.file_path}\n")
            f.write(f"Shape: {self.df.shape}\n")
            f.write(f"Target: {self.target}\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Categorical variance threshold: {self.cat_var_threshold}\n")
            f.write(f"Categorical cardinality threshold: {self.cat_card_threshold}\n\n")
            f.write("-- Summary --\n")
            for line in self.report:
                f.write(line + "\n")
        logging.info("✅ Full EDA completed. Report saved as 'eda_master_report.txt'")

    # -------------------------------------------------------------------------
    # 31. Interpretation guidance printer (ASCII only)
    # -------------------------------------------------------------------------
    def _print_guidance(self, analysis_name):
        """Enhanced interpretation guidance with ML decision focus (ASCII only)"""
        guidance = {
            'missing_analysis': """
            MISSING DATA ANALYSIS

            What to Look For:
            * Columns with high missing %
            * Patterns (random vs structured missingness)

            Why It Matters:
            * Missing != random -> can carry signal
            * High missing -> unstable model behavior

            Red Flags:
            * >40% missing -> consider dropping
            * Entire rows missing multiple columns -> systemic issue

            What To Do:
            * <5% -> simple imputation (mean/median)
            * 5-30% -> model-based or grouped imputation
            * Add missing indicator (very powerful!)
            * If missing correlates with target -> KEEP IT as feature

            Pro Tip:
            Missing data is often a hidden feature, not just a problem.
        """,

            'target_analysis': f"""
            TARGET ANALYSIS ({self.task.upper()})

            What to Look For:
            * Distribution shape
            * Class imbalance
            * Outliers

            Why It Matters:
            * Target drives entire model learning

            Red Flags:
            {"* Heavy skew -> model bias\n* Extreme outliers -> unstable regression" if self.task=='regression' else
            "* Imbalance ratio > 3 -> biased predictions\n* Minority class <5% -> hard to learn"}

            What To Do:
            {"* Apply log/box-cox transform\n* Clip extreme values" if self.task=='regression' else
            "* Use class weights / SMOTE\n* Use AUC, F1 instead of accuracy"}

            Pro Tip:
            If target is messy -> model will ALWAYS struggle.
        """,

            'univariate_analysis': """
            UNIVARIATE ANALYSIS

            What to Look For:
            * Distribution shape
            * Skewness
            * Outliers
            * Rare categories

            Why It Matters:
            * Bad distributions -> bad model assumptions

            Red Flags:
            * |skew| > 1 -> strong skew
            * Too many zeros -> zero inflation
            * Rare categories (<1%)

            What To Do:
            * Log transform skewed features
            * Bin heavy-tailed distributions
            * Group rare categories -> "Other"

            Pro Tip:
            Tree models handle skew -> linear models don't.
        """,

            'bivariate_analysis': """
            BIVARIATE ANALYSIS

            What to Look For:
            * Relationships between features
            * Clusters / separability

            Why It Matters:
            * Reveals interactions and redundancy

            Red Flags:
            * Strong correlation (>0.8) -> redundancy
            * No visible separation -> weak feature

            What To Do:
            * Drop one of highly correlated features
            * Engineer interactions for non-linear patterns

            Pro Tip:
            Models LOVE interactions you discover here.
        """,

            'target_vs_features': """
            FEATURE -> TARGET RELATIONSHIP

            What to Look For:
            * Clear trend / separation
            * Non-linear patterns

            Why It Matters:
            * Direct measure of predictive power

            Red Flags:
            * Completely random scatter -> useless feature
            * Overlapping distributions -> weak signal

            What To Do:
            * Strong pattern -> keep feature
            * Non-linear -> try tree models or transformations
            * No pattern -> consider dropping

            Pro Tip:
            If human eye sees no pattern -> model likely won't either.
        """,

            'correlation_with_target': """
            CORRELATION WITH TARGET

            What to Look For:
            * High absolute correlation

            Why It Matters:
            * Indicates linear predictive strength

            Red Flags:
            * >0.95 -> possible leakage
            * Near 0 -> not useful (linear)

            What To Do:
            * Keep high correlation features
            * Combine with MI for non-linear insights

            Pro Tip:
            Low correlation != useless (could be non-linear).
        """,

            'outlier_analysis': """
            OUTLIER ANALYSIS

            What to Look For:
            * Extreme values

            Why It Matters:
            * Can distort model training

            Red Flags:
            * >5% outliers -> heavy-tailed distribution

            What To Do:
            * Cap (winsorize)
            * Use RobustScaler
            * Remove only if clearly errors

            Pro Tip:
            Outliers are often the MOST valuable business signals.
        """,

            'vif_analysis': """
            MULTICOLLINEARITY (VIF)

            What to Look For:
            * VIF > 5 or 10

            Why It Matters:
            * Redundant features confuse models

            Red Flags:
            * VIF > 10 -> strong multicollinearity

            What To Do:
            * Drop redundant features
            * Use PCA if needed

            Pro Tip:
            Tree models don't care, linear models DO.
        """,

            'mutual_information_analysis': """
            MUTUAL INFORMATION

            What to Look For:
            * High MI values

            Why It Matters:
            * Captures NON-LINEAR relationships

            Red Flags:
            * MI = 0 -> useless feature

            What To Do:
            * Use MI for feature selection
            * Combine with correlation

            Pro Tip:
            MI finds signals correlation completely misses.
        """,

            'leakage_detection': """
            DATA LEAKAGE DETECTION

            What to Look For:
            * Extremely high correlation
            * ID-like columns

            Why It Matters:
            * Leakage = fake performance

            Red Flags:
            * Correlation > 0.95
            * Unique per row columns

            What To Do:
            * DROP leakage features immediately

            Pro Tip:
            If model accuracy looks "too good" -> leakage exists.
        """,

            'drift_detection': """
            DATA DRIFT DETECTION

            What to Look For:
            * Distribution differences (p < 0.05)

            Why It Matters:
            * Model trained on old data fails on new data

            Red Flags:
            * Strong drift in key features

            What To Do:
            * Retrain model
            * Use time-based validation

            Pro Tip:
            Drift is the #1 reason models fail in production.
        """,

            'feature_stability_analysis': """
            FEATURE STABILITY

            What to Look For:
            * Consistency across splits

            Why It Matters:
            * Unstable features -> unreliable model

            Red Flags:
            * Distribution shifts across splits

            What To Do:
            * Drop unstable features
            * Use robust transformations

            Pro Tip:
            Stable features > highly predictive unstable ones.
        """,

            'single_feature_performance': """
            SINGLE FEATURE PERFORMANCE

            What to Look For:
            * Individual predictive strength

            Why It Matters:
            * Identifies strong standalone features

            Red Flags:
            * Very low score -> weak feature

            What To Do:
            * Keep strong features
            * Combine weak ones for interactions

            Pro Tip:
            Weak alone != useless (interactions matter).
        """,

            'monotonicity_analysis': """
            MONOTONIC RELATIONSHIP

            What to Look For:
            * Smooth increasing/decreasing trend

            Why It Matters:
            * Easy for models to learn

            Red Flags:
            * Highly irregular relationship

            What To Do:
            * Bin feature if monotonic
            * Use linear models confidently

            Pro Tip:
            Monotonic features are gold in finance/insurance.
        """,

            'rare_category_analysis': """
            RARE CATEGORY ANALYSIS

            What to Look For:
            * Categories <1%

            Why It Matters:
            * Causes overfitting

            Red Flags:
            * Many rare categories

            What To Do:
            * Group into "Other"
            * Use frequency encoding

            Pro Tip:
            Rare categories = noise 80% of the time.
        """,

            'feature_recommendation_engine': """
            FEATURE RECOMMENDATION ENGINE

            What it Does:
            * Combines all signals

            Why It Matters:
            * Converts EDA -> ACTION

            Output You Should Use:
            * Drop list
            * Transform list
            * Keep list

            Pro Tip:
            This is where EDA becomes MODEL STRATEGY.
        """
        }

        print(guidance.get(analysis_name, "\n[Interpretation] No guidance available."))

    # -------------------------------------------------------------------------
    # NEW: Generate a comprehensive text report (no images)
    # -------------------------------------------------------------------------
    def generate_text_report(self, analyses=None, output_file="ml_analysis_report.txt"):
        """
        Generate a detailed text report of all EDA analyses.
        If analyses is None, runs all available analyses.
        If analyses is a list, runs only those and includes them in the report.
        The report is saved to output_file.
        """
        # Run the requested analyses if not already run (we re-run them to ensure fresh data)
        if analyses is None:
            analyses = self.AVAILABLE_ANALYSES
        elif isinstance(analyses, str):
            analyses = [analyses]

        # Temporarily disable plotting to avoid showing windows during report generation
        plt.ioff()
        original_show = self.show_guidance
        self.show_guidance = False  # we will add guidance text ourselves

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("MACHINE LEARNING MODEL PREPARATION – COMPREHENSIVE EDA REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Dataset: {self.file_path}")
        report_lines.append(f"Shape: {self.df.shape}")
        report_lines.append(f"Target column: {self.target}")
        report_lines.append(f"Task type: {self.task}")
        report_lines.append(f"Categorical variance threshold: {self.cat_var_threshold}")
        report_lines.append(f"Categorical cardinality threshold: {self.cat_card_threshold}")
        report_lines.append(f"Date & time: {pd.Timestamp.now()}")
        report_lines.append("\n")

        for analysis in analyses:
            if analysis not in self.AVAILABLE_ANALYSES:
                logging.warning(f"Skipping unknown analysis: {analysis}")
                continue

            # Add header for this analysis
            report_lines.append("-" * 80)
            report_lines.append(f"ANALYSIS: {analysis.upper()}")
            report_lines.append("-" * 80)

            # Compute and write the description for this analysis
            desc = self._describe_analysis(analysis)
            report_lines.extend(desc.split('\n'))

            # Add interpretation guidance (from _print_guidance)
            guidance_text = self._get_guidance_text(analysis)
            if guidance_text:
                report_lines.append("\n--- INTERPRETATION & GUIDANCE ---")
                report_lines.extend(guidance_text.split('\n'))

            report_lines.append("\n")

        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        self.show_guidance = original_show
        plt.ion()  # restore interactive mode
        logging.info(f"✅ Text report saved to {output_file}")

    def _describe_analysis(self, analysis_name):
        """
        Generate a textual description for a given analysis.
        This method uses the data and existing computed statistics to
        produce a detailed description of what would be seen in the plots.
        """
        buf = io.StringIO()

        if analysis_name == 'missing_analysis':
            missing = self.df.isnull().sum()
            missing = missing[missing > 0]
            buf.write("MISSING DATA ANALYSIS\n\n")
            if missing.empty:
                buf.write("✅ No missing values found.\n")
            else:
                buf.write("Missing values detected. The following columns have missing data:\n")
                for col, count in missing.items():
                    pct = 100 * count / len(self.df)
                    buf.write(f"  * {col}: {count} ({pct:.2f}%)\n")
                buf.write("\nPlots that would be shown:\n")
                buf.write("  1. Heatmap: Visualizes missing data patterns across rows and columns.\n")
                buf.write("  2. Bar chart: Percentage missing per column, sorted.\n")
                buf.write("\nKey insights:\n")
                high_missing = [col for col, pct in (100 * missing / len(self.df)).items() if pct > 40]
                if high_missing:
                    buf.write(f"  * Columns with >40% missing: {', '.join(high_missing)} – consider dropping or treating as signal.\n")
                if any(100 * missing / len(self.df) > 20):
                    buf.write("  * Several columns have >20% missing; imputation strategy must be carefully chosen.\n")
                else:
                    buf.write("  * Missingness is relatively low; simple imputation may suffice.\n")

        elif analysis_name == 'target_analysis':
            buf.write("TARGET VARIABLE ANALYSIS\n\n")
            if self.task == 'regression':
                stats_desc = self.df[self.target].describe()
                buf.write(f"Summary statistics:\n{stats_desc}\n")
                buf.write("\nPlots that would be shown:\n")
                buf.write("  1. Histogram with KDE: Shows distribution shape, skewness, and presence of outliers.\n")
                buf.write("  2. Boxplot: Identifies extreme values and spread.\n")
                skew = self.df[self.target].skew()
                buf.write(f"\nKey insights:\n  * Skewness = {skew:.3f}")
                if abs(skew) > 1:
                    buf.write(" -> Highly skewed; consider log or Box-Cox transformation.\n")
                else:
                    buf.write(" -> Moderate skew; transformation may still help.\n")
                # Check outliers
                q1 = self.df[self.target].quantile(0.25)
                q3 = self.df[self.target].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = self.df[(self.df[self.target] < lower) | (self.df[self.target] > upper)]
                buf.write(f"  * Outliers (IQR method): {len(outliers)} rows ({100*len(outliers)/len(self.df):.2f}%)\n")
            else:
                vc = self.df[self.target].value_counts()
                buf.write("Class distribution:\n")
                for cls, cnt in vc.items():
                    pct = 100 * cnt / len(self.df)
                    buf.write(f"  {cls}: {cnt} ({pct:.2f}%)\n")
                imbalance = vc.max() / vc.min()
                buf.write(f"\nImbalance ratio (max/min): {imbalance:.2f}\n")
                buf.write("\nPlots that would be shown:\n")
                buf.write("  1. Bar chart: Absolute class counts.\n")
                buf.write("  2. Pie chart: Relative class proportions.\n")
                if imbalance > 3:
                    buf.write("  ⚠️ Significant class imbalance detected. Consider resampling (SMOTE) or using class weights.\n")
                else:
                    buf.write("  ✅ Classes are relatively balanced.\n")

        elif analysis_name == 'univariate_analysis':
            buf.write("UNIVARIATE ANALYSIS (Individual Features)\n\n")
            if self.num_cols:
                buf.write("Numerical features:\n")
                for col in self.num_cols:
                    skew = self.df[col].skew()
                    buf.write(f"  * {col}: skewness = {skew:.3f}")
                    if abs(skew) > 1:
                        buf.write(" (highly skewed)\n")
                    else:
                        buf.write("\n")
                buf.write("\nPlots that would be shown:\n")
                buf.write("  - Histograms with KDE for each numerical feature.\n")
                buf.write("  - Boxplots to identify outliers.\n")
            if self.cat_cols:
                buf.write("\nCategorical features:\n")
                for col in self.cat_cols:
                    uniq = self.df[col].nunique()
                    buf.write(f"  * {col}: {uniq} unique categories")
                    if uniq > 50:
                        buf.write(" (high cardinality)\n")
                    else:
                        buf.write("\n")
                buf.write("\nPlots that would be shown:\n")
                buf.write("  - Bar charts of category frequencies for each categorical feature.\n")
            buf.write("\nKey insights:\n")
            high_card = [col for col in self.cat_cols if self.df[col].nunique() > 50]
            if high_card:
                buf.write(f"  * High-cardinality categoricals: {', '.join(high_card)} – consider target encoding or frequency encoding.\n")
            zero_var = [col for col in self.num_cols if self.df[col].var() == 0]
            if zero_var:
                buf.write(f"  * Zero-variance numericals: {', '.join(zero_var)} – these provide no information; drop them.\n")

        elif analysis_name == 'correlation_with_target':
            if self.task == 'regression' and self.num_cols:
                corr = self.df[self.num_cols].corrwith(self.df[self.target]).abs().sort_values(ascending=False)
                buf.write("CORRELATION WITH TARGET\n\n")
                buf.write("Top correlations (absolute value):\n")
                for col, val in corr.head(10).items():
                    buf.write(f"  * {col}: {val:.4f}\n")
                buf.write("\nPlots that would be shown:\n")
                buf.write("  - Bar chart of correlation coefficients.\n")
                high_corr = corr[corr > 0.95]
                if not high_corr.empty:
                    buf.write(f"\n⚠️ Potential leakage: {', '.join(high_corr.index)} have |corr| > 0.95. Consider removing if suspicious.\n")
            else:
                buf.write("Correlation analysis is only applicable for regression tasks with numeric features.\n")

        elif analysis_name == 'mutual_information_analysis':
            try:
                X = self.df[self.num_cols + self.cat_cols].copy()
                y = self.df[self.target]
                for col in self.cat_cols:
                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                X_clean = X.dropna()
                y_clean = y.loc[X_clean.index]
                if self.task == 'regression':
                    mi = mutual_info_regression(X_clean, y_clean, random_state=42)
                else:
                    mi = mutual_info_classif(X_clean, y_clean, random_state=42)
                mi_df = pd.DataFrame({'feature': X.columns, 'mutual_info': mi})
                mi_df = mi_df.sort_values('mutual_info', ascending=False)
                buf.write("MUTUAL INFORMATION WITH TARGET\n\n")
                buf.write("Top features by mutual information:\n")
                for _, row in mi_df.head(10).iterrows():
                    buf.write(f"  * {row['feature']}: {row['mutual_info']:.4f}\n")
                buf.write("\nPlots that would be shown:\n")
                buf.write("  - Bar chart of mutual information scores.\n")
                buf.write("\nKey insights:\n")
                if mi_df.iloc[0]['mutual_info'] < 0.01:
                    buf.write("  * All MI scores are very low – features may have little predictive power.\n")
                else:
                    buf.write("  * High MI features capture non-linear relationships not seen in correlation.\n")
            except Exception as e:
                buf.write(f"Could not compute mutual information: {e}\n")

        else:
            # Generic fallback
            buf.write(f"Analysis '{analysis_name}' is part of the comprehensive EDA suite.\n")
            buf.write("It examines aspects such as feature interactions, drift, encoding decisions, etc.\n")
            buf.write("For a detailed narrative, refer to the interpretation guidance below.\n")

        return buf.getvalue()

    def _get_guidance_text(self, analysis_name):
        """
        Retrieve the interpretation guidance text for a given analysis.
        This reuses the content from _print_guidance but returns it as a string.
        """
        # Reuse the ASCII guidance from _print_guidance – we'll call the method and capture its output
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        self._print_guidance(analysis_name)
        guidance_output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        return guidance_output.strip()

    # -------------------------------------------------------------------------
    # 32. Orchestration methods
    # -------------------------------------------------------------------------
    def list_analyses(self):
        print("Available analyses (call as methods):")
        for name in self.AVAILABLE_ANALYSES:
            print(f"  - {name}")

    def run(self, analyses=None):
        if analyses is None:
            analyses = self.AVAILABLE_ANALYSES
        elif isinstance(analyses, str):
            analyses = [analyses]
        for name in analyses:
            if name not in self.AVAILABLE_ANALYSES:
                logging.warning(f"Unknown analysis '{name}'. Skipping. Available: {self.AVAILABLE_ANALYSES}")
                continue
            method = getattr(self, name, None)
            if method is not None:
                logging.info(f"Running analysis: {name}")
                method()
            else:
                logging.warning(f"Method '{name}' not found. Skipping.")

    def run_all(self):
        self.run()


# ============================================================================
# Usage Example
# ============================================================================
if __name__ == "__main__":
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(),
                                      logging.FileHandler('eda.log')])

    file_path = input("Enter dataset path: ")
    target = input("Enter target column: ")
    task = input("Task type no | regression: 1 | binary classification: 2 | multi classification: 3: ")

    guidance = input("Show interpretation guidance after each analysis? (y/n): ").strip().lower()
    show_guidance = guidance == 'y'

    eda = ComprehensiveEDA(file_path, target, task, show_guidance=show_guidance)

    print("\nChoose how to run:")
    print("1. Run all analyses (plots)")
    print("2. Run specific analyses (plots)")
    print("3. Generate text report only (no plots)")
    choice = input("Enter choice (1, 2, or 3): ").strip()

    if choice == '1':
        eda.run_all()
        eda.save_report()
    elif choice == '2':
        eda.list_analyses()
        selected = input("Enter analysis names separated by commas (e.g., missing_analysis,target_analysis): ").strip()
        if selected:
            selected_list = [s.strip() for s in selected.split(',') if s.strip()]
            eda.run(selected_list)
            eda.save_report()
        else:
            print("No analyses selected. Exiting.")
    elif choice == '3':
        # Generate only the text report without showing plots
        eda.generate_text_report()
    else:
        print("Invalid choice.")