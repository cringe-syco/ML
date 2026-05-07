import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import warnings
import json
from datetime import datetime


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
        'data_summary',
        'correlation_heatmap',
        'time_series_analysis',
        'data_quality_checks'
    ]

    def __init__(self, file_path, target, task_no, 
                categorical_variance_threshold=0.01,
                categorical_cardinality_threshold=None,
                show_guidance=False,
                show_plots=True,
                z_threshold=3,
                leakage_corr_threshold=0.95,
                encoding_cardinality_threshold=20,
                max_categories_interaction=20,
                top_n_pairs=10,
                mi_plot_top_n=10,
                sample_plot_rows=10000,
                time_col=None,
                report_filename='eda_master_report.txt',
                report_format='txt',
                append_timestamp=False,
                test_df=None,
                outlier_method='zscore'):
        """
        Parameters:
        -----------
        file_path : str
            Path to the dataset (CSV, Excel, or Parquet).
        target : str
            Name of the target column.
        task_no : str
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
        self.show_plots = show_plots
        self.z_threshold = z_threshold
        self.leakage_corr_threshold = leakage_corr_threshold
        self.encoding_cardinality_threshold = encoding_cardinality_threshold
        self.max_categories_interaction = max_categories_interaction
        self.top_n_pairs = top_n_pairs
        self.mi_plot_top_n = mi_plot_top_n
        self.sample_plot_rows = sample_plot_rows
        self.time_col = time_col
        self.report_filename = report_filename
        self.report_format = report_format
        self.append_timestamp = append_timestamp
        self.test_df = test_df
        self.outlier_method = outlier_method
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

        # Keep an immutable "clean" copy for analyses (do not mutate after init)
        self.df_original = self.df.copy()

        # Setup logging (only if no handlers exist)
        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO,
                                format='%(asctime)s - %(levelname)s - %(message)s',
                                handlers=[logging.StreamHandler(),
                                        logging.FileHandler('eda.log')])
        logging.info(f"EDA initialized: target={target}, task={self.task}, shape={self.df.shape}")

    # -------------------------------------------------------------------------
    # Helper utilities
    # -------------------------------------------------------------------------
    def _get_df(self, copy=True):
        """Return a working dataframe for analyses (never mutate self.df_original)."""
        return self.df_original.copy() if copy else self.df_original

    def _maybe_sample(self, df, n=None):
        n = self.sample_plot_rows if n is None else n
        if len(df) > n:
            logging.info(f"Sampling {n} rows for plotting due to large dataset ({len(df)} rows).")
            return df.sample(n=n, random_state=42)
        return df

    def _ordinal_encode_df(self, df, columns):
        """Encode categorical columns with OrdinalEncoder; fallback to factorize on error."""
        if not columns:
            return df, None
        try:
            from sklearn.preprocessing import OrdinalEncoder
            enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            df[columns] = enc.fit_transform(df[columns].astype(str))
            return df, enc
        except Exception:
            logging.warning("OrdinalEncoder failed; falling back to pandas.factorize per column.")
            for col in columns:
                df[col], _ = pd.factorize(df[col].astype(str))
            return df, None

    # -------------------------------------------------------------------------
    # 1. Data loading
    # -------------------------------------------------------------------------
    def _load_data(self):
        """Load dataset from CSV/Excel/Parquet file."""
        try:
            if self.file_path.endswith('.csv'):
                try:
                    return pd.read_csv(self.file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    logging.warning(f"utf-8 decode failed for {self.file_path}; retrying with latin-1")
                    return pd.read_csv(self.file_path, encoding='latin-1')
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
                if pd.to_datetime(sample, errors='coerce') is pd.NaT:
                    continue
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                logging.info(f"Column '{col}' converted to datetime (coercing errors).")
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

        # Warn about mixed-type object columns
        for col in list(cat):
            try:
                type_counts = self.df[col].dropna().map(lambda x: type(x)).nunique()
                if type_counts > 1:
                    msg = f"Column '{col}' contains mixed types ({type_counts}); treating as categorical."
                    logging.warning(msg)
                    self.report.append(msg)
            except Exception:
                continue

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
        df = self._get_df()
        missing = df.isnull().sum()
        missing = missing[missing > 0]

        if missing.empty:
            logging.info("No missing values found.")
            self.report.append("No missing values found.")
            if self.show_guidance:
                print("\n[Interpretation] No missing values – good data quality.")
            return

        logging.info(f"Missing values:\n{missing}")
        self.report.append(f"Missing values per column:\n{missing}")

        sample_df = self._maybe_sample(df)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(sample_df.isnull(), cbar=False, yticklabels=False, cmap='viridis', ax=ax)
        ax.set_title("Missing Data Heatmap")
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

        missing_pct = (missing / len(df)) * 100
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        missing_pct.sort_values().plot(kind='bar', ax=ax2)
        ax2.set_ylabel('Percentage Missing')
        ax2.set_title('Missing Data Percentage by Column')
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig2)

        plt.close('all')
        if self.show_guidance:
            self._print_guidance('missing_analysis')

    # -------------------------------------------------------------------------
    # 5. Target variable analysis
    # -------------------------------------------------------------------------
    def target_analysis(self):
        """Analyse target distribution (histogram + boxplot for regression;
           bar + pie for classification)."""
        logging.info("===== TARGET ANALYSIS =====")
        df = self._get_df()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax1, ax2 = axes

        if self.task == 'regression':
            sns.histplot(df[self.target], kde=True, ax=ax1)
            ax1.set_title(f"Distribution of {self.target}")
            sns.boxplot(x=df[self.target], ax=ax2)
            ax2.set_title(f"Boxplot of {self.target}")
            stats = df[self.target].describe()
            logging.info(f"Target statistics:\n{stats}")
            self.report.append(f"Target statistics:\n{stats}")
        else:
            vc = df[self.target].value_counts()
            sns.barplot(x=vc.index, y=vc.values, ax=ax1)
            ax1.set_title(f"Class Distribution of {self.target}")
            ax1.set_xlabel('Class')
            ax1.set_ylabel('Count')
            for i, val in enumerate(vc.values):
                ax1.text(i, val + 0.5, str(val), ha='center')
            ax2.pie(vc.values, labels=vc.index, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Class Proportions')
            imbalance = vc.max() / vc.min() if vc.min() > 0 else float('inf')
            logging.info(f"Class counts:\n{vc}")
            logging.info(f"Imbalance Ratio (max/min): {imbalance:.2f}")
            self.report.append(f"Class counts:\n{vc}")
            self.report.append(f"Imbalance Ratio: {imbalance:.2f}")

        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        if self.show_guidance:
            self._print_guidance('target_analysis')
        plt.close('all')

    # -------------------------------------------------------------------------
    # 6. Univariate analysis
    # -------------------------------------------------------------------------
    def univariate_analysis(self):
        """Plot histograms + boxplots for numeric, bar charts for categorical."""
        logging.info("===== UNIVARIATE ANALYSIS =====")
        df = self._get_df()
        if self.num_cols:
            n_cols = min(3, len(self.num_cols))
            n_rows = (len(self.num_cols) + n_cols - 1) // n_cols
            df_plot = self._maybe_sample(df)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            for i, col in enumerate(self.num_cols):
                sns.histplot(df_plot[col], kde=True, ax=axes[i], color='skyblue')
                axes[i].set_title(f"{col} (skew={df[col].skew():.2f})")
                self.report.append(f"{col} skewness: {df[col].skew():.2f}")
            for i in range(len(self.num_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)

            fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes2 = axes2.flatten() if n_rows * n_cols > 1 else [axes2]
            for i, col in enumerate(self.num_cols):
                sns.boxplot(x=df_plot[col], ax=axes2[i], color='lightgreen')
                axes2[i].set_title(col)
            for i in range(len(self.num_cols), len(axes2)):
                axes2[i].set_visible(False)
            plt.tight_layout()
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig2)

        if self.cat_cols:
            df_plot = self._maybe_sample(df)
            n_cols = min(3, len(self.cat_cols))
            n_rows = (len(self.cat_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
            for i, col in enumerate(self.cat_cols):
                vc = df_plot[col].value_counts()
                sns.barplot(x=vc.index, y=vc.values, ax=axes[i], palette='viridis')
                axes[i].set_title(f"{col} (n_unique={len(vc)})")
                axes[i].tick_params(axis='x', rotation=45)
                for j, val in enumerate(vc.values):
                    axes[i].text(j, val + 0.5, str(val), ha='center', fontsize=8)
            for i in range(len(self.cat_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)

        if self.show_guidance:
            self._print_guidance('univariate_analysis')
        plt.close('all')

    # -------------------------------------------------------------------------
    # 7. Bivariate analysis
    # -------------------------------------------------------------------------
    def bivariate_analysis(self):
        """Pairplot for numeric; boxplots for categorical vs numeric."""
        logging.info("===== BIVARIATE ANALYSIS =====")
        df = self._get_df()
        if len(self.num_cols) >= 2:
            if len(self.num_cols) > 10:
                logging.warning("Too many numerical columns (>10). Skipping pairplot to avoid memory issues.")
            else:
                df_pair = self._maybe_sample(df[self.num_cols])
                g = sns.pairplot(df_pair, diag_kind='kde')
                plt.suptitle('Pairplot of Numerical Features', y=1.02)
                if self.show_plots:
                    plt.show()
                else:
                    plt.close(g.fig)

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

            df_plot = self._maybe_sample(df)
            for i, cat in enumerate(self.cat_cols):
                for j, num in enumerate(self.num_cols):
                    ax = axes[i][j]
                    sns.boxplot(x=df_plot[cat], y=df_plot[num], ax=ax)
                    ax.set_title(f"{cat} vs {num}")
                    ax.tick_params(axis='x', rotation=45)
                    if i == n_rows - 1:
                        ax.set_xlabel(cat)
                    if j == 0:
                        ax.set_ylabel(num)
            plt.tight_layout()
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)

        if self.show_guidance:
            self._print_guidance('bivariate_analysis')
        plt.close('all')

    # -------------------------------------------------------------------------
    # 8. Target vs features
    # -------------------------------------------------------------------------
    def target_vs_features(self):
        """Numerical vs target (scatter/box); categorical vs target (stacked bar) in subplots."""
        logging.info("===== TARGET vs FEATURES =====")
        df = self._get_df()
        if self.num_cols:
            n_cols = min(3, len(self.num_cols))
            n_rows = (len(self.num_cols) + n_cols - 1) // n_cols
            df_plot = self._maybe_sample(df)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.num_cols):
                if self.task == 'regression':
                    sns.scatterplot(x=df_plot[col], y=df_plot[self.target], ax=axes[i], alpha=0.5)
                    axes[i].set_title(f"{col} vs {self.target}")
                else:
                    if df[self.target].nunique() > 20:
                        logging.warning(f"Target has {df[self.target].nunique()} unique values. Boxplot may be crowded.")
                    sns.boxplot(x=df_plot[self.target], y=df_plot[col], ax=axes[i])
                    axes[i].set_title(f"{col} vs {self.target}")
                    axes[i].tick_params(axis='x', rotation=45)
            for i in range(len(self.num_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)

        if self.cat_cols and self.task != 'regression':
            n_cols = min(3, len(self.cat_cols))
            n_rows = (len(self.cat_cols) + n_cols - 1) // n_cols
            df_plot = self._get_df()
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.cat_cols):
                crosstab = pd.crosstab(df_plot[col], df_plot[self.target], normalize='index') * 100
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
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)

        if self.show_guidance:
            self._print_guidance('target_vs_features')
        plt.close('all')

    # -------------------------------------------------------------------------
    # 9. Correlation with target (regression only)
    # -------------------------------------------------------------------------
    def correlation_with_target(self):
        df = self._get_df()
        if self.task == 'regression' and self.num_cols:
            logging.info("===== FEATURE IMPORTANCE (Correlation) =====")
            corr = df[self.num_cols + [self.target]].corr()[self.target].drop(self.target).sort_values(ascending=False)
            logging.info(f"Correlation with target:\n{corr}")
            self.report.append(f"Correlation with target:\n{corr}")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x=corr.values, y=corr.index, palette='coolwarm', ax=ax)
            ax.set_title(f'Correlation with Target: {self.target}')
            ax.set_xlabel('Correlation Coefficient')
            plt.tight_layout()
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)
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
            df = self._get_df()
            logging.info("===== GROUPED ANALYSIS (Categorical vs Target Mean) =====")
            n_cols = min(3, len(self.cat_cols))
            n_rows = (len(self.cat_cols) + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
            axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

            for i, col in enumerate(self.cat_cols):
                grouped = df.groupby(col)[self.target].mean().sort_values()
                sns.barplot(x=grouped.index, y=grouped.values, ax=axes[i])
                axes[i].set_title(f"{col} -> Mean {self.target}")
                axes[i].tick_params(axis='x', rotation=45)
                for j, val in enumerate(grouped.values):
                    axes[i].text(j, val + 0.01, f"{val:.2f}", ha='center', fontsize=8)
            for i in range(len(self.cat_cols), len(axes)):
                axes[i].set_visible(False)
            plt.tight_layout()
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig)
            if self.show_guidance:
                self._print_guidance('grouped_analysis')
            plt.close('all')
        else:
            if self.show_guidance and self.task == 'regression':
                print("[Interpretation] No categorical features available for grouped analysis.")

    # -------------------------------------------------------------------------
    # 11. Outlier detection (Z-score + boxplots)
    # -------------------------------------------------------------------------
    def outlier_analysis(self, method=None):
        method = method or self.outlier_method
        if not self.num_cols:
            if self.show_guidance:
                print("[Interpretation] No numerical columns to analyze for outliers.")
            return
        logging.info("===== OUTLIERS DETECTION =====")
        n_cols = min(3, len(self.num_cols))
        n_rows = (len(self.num_cols) + n_cols - 1) // n_cols
        df = self._get_df()
        df_plot = self._maybe_sample(df)
        fig, axes = plt.subplots(n_cols if n_rows == 1 else n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]

        for i, col in enumerate(self.num_cols):
            if df[col].std() == 0 or np.isnan(df[col].std()):
                outliers = 0
                logging.info(f"{col}: constant column, no outliers.")
            else:
                if method == 'iqr':
                    q1 = df[col].quantile(0.25)
                    q3 = df[col].quantile(0.75)
                    iqr = q3 - q1
                    outliers = ((df[col] < (q1 - 1.5 * iqr)) | (df[col] > (q3 + 1.5 * iqr))).sum()
                    logging.info(f"{col}: {outliers} outliers (IQR method)")
                    self.report.append(f"{col}: {outliers} outliers (IQR method)")
                else:
                    z = np.abs((df[col] - df[col].mean()) / (df[col].std() if df[col].std() != 0 else 1))
                    outliers = int((z > self.z_threshold).sum())
                    logging.info(f"{col}: {outliers} outliers (Z>{self.z_threshold})")
                    self.report.append(f"{col}: {outliers} outliers (Z>{self.z_threshold})")
            sns.boxplot(x=df_plot[col], ax=axes[i])
            axes[i].set_title(f"{col} (outliers={outliers})")
        for i in range(len(self.num_cols), len(axes)):
            axes[i].set_visible(False)
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        if self.show_guidance:
            self._print_guidance('outlier_analysis')

    # -------------------------------------------------------------------------
    # 12. Duplicate analysis
    # -------------------------------------------------------------------------
    def duplicate_analysis(self):
        df = self._get_df()
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            logging.info(f"Found {duplicates} duplicate rows.")
            self.report.append(f"Duplicate rows: {duplicates}")
            dup_examples = df[df.duplicated(keep=False)].head(5)
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
        df = self._get_df()
        X = df[self.num_cols].dropna()
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        vif_data = vif_data.sort_values("VIF", ascending=False)
        logging.info(f"VIF scores:\n{vif_data}")
        self.report.append(f"VIF scores:\n{vif_data}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=vif_data, x="VIF", y="feature", palette="rocket", ax=ax)
        ax.axvline(x=5, color='red', linestyle='--', label='Threshold (5)')
        ax.axvline(x=10, color='orange', linestyle='--', label='High (10)')
        ax.set_title("Variance Inflation Factor (VIF)")
        ax.legend()
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        if self.show_guidance:
            self._print_guidance('vif_analysis')
        plt.close('all')

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
        df = self._get_df()
        X = df[self.num_cols + self.cat_cols].copy()
        y = df[self.target]
        if self.cat_cols:
            X, _ = self._ordinal_encode_df(X, self.cat_cols)
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
        fig, ax = plt.subplots(figsize=(10, 6))
        top_plot = mi_df.head(self.mi_plot_top_n)
        sns.barplot(data=top_plot, x='mutual_info', y='feature', palette='viridis', ax=ax)
        ax.set_title(f"Mutual Information with Target ({self.task})")
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        if self.show_guidance:
            self._print_guidance('mutual_information_analysis')
        plt.close('all')

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
        df = self._get_df()
        if self.num_cols:
            logging.info("Normality tests (Shapiro-Wilk):")
            for col in self.num_cols[:5]:
                sample = df[col].dropna()
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
                    categories = df[cat].dropna().unique()
                    for val in categories:
                        group_data = df[self.target][df[cat] == val].dropna()
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
                    target_classes = df[self.target].dropna().unique()
                    for cls in target_classes:
                        group_data = df[num][df[self.target] == cls].dropna()
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
                    ct = pd.crosstab(df[cat], df[self.target])
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
        df = self._get_df()
        threshold = target_correlation_threshold or self.leakage_corr_threshold
        if self.task == 'regression' and self.num_cols:
            corr = df[self.num_cols].corrwith(df[self.target]).abs()
            high_corr = corr[corr > threshold]
            if not high_corr.empty:
                logging.info(f"Potential leakage: features with |corr| > {threshold}:\n{high_corr}")
                self.report.append(f"Potential leakage (high correlation): {high_corr.index.tolist()}")
            else:
                logging.info("No features with suspiciously high correlation to target.")
                self.report.append("No high-correlation leakage detected.")
        for col in df.columns:
            try:
                if df[col].nunique() == len(df) and (pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_object_dtype(df[col])):
                    logging.info(f"Column '{col}' appears to be a unique identifier (potential leakage if used as feature).")
                    self.report.append(f"Unique identifier column: {col}")
            except Exception:
                continue
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
        df = self._get_df()
        num_cols = self.num_cols
        if len(num_cols) < 2:
            logging.info("Need at least two numeric features for interaction analysis.")
            return
        # limit pairwise work to avoid explosion
        max_cols = min(len(num_cols), 30)
        if len(num_cols) > max_cols:
            logging.warning(f"Limiting numeric features to first {max_cols} for interaction scan.")
            num_cols = num_cols[:max_cols]

        scores = []
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                c1, c2 = num_cols[i], num_cols[j]
                inter = df[c1] * df[c2]
                inter_clean = inter.dropna()
                y_clean = df.loc[inter_clean.index, self.target]
                if len(inter_clean) < 10:
                    continue
                if self.task == 'regression':
                    mi = mutual_info_regression(inter_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
                else:
                    mi = mutual_info_classif(inter_clean.values.reshape(-1, 1), y_clean, random_state=42)[0]
                scores.append((f"{c1}*{c2}", mi))

        if not scores:
            logging.info("No interaction scores computed (insufficient data).")
            return

        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:top_n]
        pairs = [s[0] for s in top_scores]
        mi_vals = [s[1] for s in top_scores]
        mi_df = pd.DataFrame({'pair': pairs, 'mi': mi_vals})
        logging.info(f"Top feature interactions (MI):\n{mi_df}")
        self.report.append(f"Top feature interactions:\n{mi_df}")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=mi_df, x='mi', y='pair', palette='mako', ax=ax)
        ax.set_title('Top Feature Interaction Terms by Mutual Information')
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        if self.show_guidance:
            self._print_guidance('feature_interactions')
        plt.close('all')

    # -------------------------------------------------------------------------
    # 18. Missing as signal
    # -------------------------------------------------------------------------
    def missing_as_signal(self):
        logging.info("===== MISSING AS SIGNAL =====")
        df_copy = self._get_df()
        missing_cols = df_copy.columns[df_copy.isnull().any()].tolist()
        if not missing_cols:
            logging.info("No missing columns to analyse.")
            if self.show_guidance:
                print("[Interpretation] No missing data to treat as signal.")
            return
        logging.info("Testing missing indicators vs target (without mutating original dataframe):")
        for col in missing_cols:
            indicator = df_copy[col].isnull().astype(int)
            if self.task == 'regression':
                corr = indicator.corr(df_copy[self.target])
                logging.info(f"Missing indicator for '{col}' correlation with target: {corr:.4f}")
                self.report.append(f"Missing indicator '{col}' correlation with target: {corr:.4f}")
            else:
                from scipy.stats import chi2_contingency
                cross = pd.crosstab(indicator, df_copy[self.target])
                if cross.shape[0] >= 2 and cross.shape[1] >= 2:
                    chi2, p, dof, expected = chi2_contingency(cross)
                    logging.info(f"Missing indicator for '{col}' vs target: chi2={chi2:.2f}, p={p:.4f}")
                    self.report.append(f"Missing indicator '{col}' vs target: p={p:.4f}")
                else:
                    logging.info(f"Missing indicator for '{col}': insufficient categories for chi-square test.")
        if self.show_guidance:
            self._print_guidance('missing_as_signal')

    # -------------------------------------------------------------------------
    # 19. Drift detection
    # -------------------------------------------------------------------------
    def drift_detection(self, test_df=None, time_col=None):
        logging.info("===== DRIFT DETECTION =====")
        df = self._get_df()
        test_df = test_df if test_df is not None else self.test_df

        if test_df is not None:
            for col in self.num_cols + self.cat_cols:
                train = df[col].dropna()
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
        else:
            # determine time column to split train/test
            time_col = time_col or self.time_col
            if time_col is None and self.dt_cols:
                time_col = self.dt_cols[0]
                logging.info(f"Auto-selected datetime column for drift detection: {time_col}")
            if time_col is not None and time_col in df.columns:
                temp = df.copy()
                temp[time_col] = pd.to_datetime(temp[time_col], errors='coerce')
                temp = temp.dropna(subset=[time_col])
                if temp.empty:
                    logging.info(f"Datetime column '{time_col}' contains no parseable dates. Skipping drift detection.")
                    return
                mid = temp[time_col].median()
                train_mask = temp[time_col] <= mid
                test_mask = temp[time_col] > mid
                for col in self.num_cols + self.cat_cols:
                    train = temp.loc[train_mask, col].dropna()
                    test = temp.loc[test_mask, col].dropna()
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
            else:
                logging.info("Drift detection requires either a test dataset or a datetime column.")
                if self.show_guidance:
                    print("[Interpretation] Provide a test_df or specify time_col to detect drift.")
        if self.show_guidance:
            self._print_guidance('drift_detection')
        plt.close('all')

    # -------------------------------------------------------------------------
    # 20. Encoding decisions
    # -------------------------------------------------------------------------
    def encoding_decisions(self, cardinality_threshold=20):
        logging.info("===== ENCODING DECISIONS =====")
        suggestions = {}
        threshold = cardinality_threshold or self.encoding_cardinality_threshold
        df = self._get_df()
        for col in self.cat_cols:
            n_unique = df[col].nunique()
            if n_unique <= 2:
                suggestions[col] = "Label encoding (binary)"
            elif n_unique <= threshold:
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
        plt.close('all')

    # -------------------------------------------------------------------------
    # 21. Categorical interactions
    # -------------------------------------------------------------------------
    def categorical_interactions(self, top_n=10, heatmaps_for_top_n=1, max_categories_per_feature=20):
        """
        Discover and visualize interactions between categorical features in classification tasks.
        
        Parameters:
        top_n : int, default=10
            Number of top feature pairs to display in bar chart.
        heatmaps_for_top_n : int, default=1
            Number of top pairs for which to generate heatmaps.
            If set > number of available pairs, it is clamped to that number.
        max_categories_per_feature : int, default=20
            If a categorical feature has more unique values than this, it is skipped.
        """
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
        df = self._get_df()
        usable_cats = [col for col in self.cat_cols if df[col].nunique() <= max_categories_per_feature]
        if len(usable_cats) < 2:
            logging.info(f"After filtering by cardinality (≤{max_categories_per_feature}), insufficient features.")
            return
        from sklearn.feature_selection import mutual_info_classif
        import warnings
        warnings.filterwarnings('ignore')
        y = df[self.target]
        scores = []
        for i in range(len(usable_cats)):
            for j in range(i+1, len(usable_cats)):
                col1, col2 = usable_cats[i], usable_cats[j]
                combined = df[col1].astype(str) + '_' + df[col2].astype(str)
                codes, _ = pd.factorize(combined)
                mi = mutual_info_classif(codes.reshape(-1,1), y, random_state=42)[0]
                scores.append((col1, col2, mi))

        if not scores:
            logging.info("No interaction scores computed.")
            return

        scores.sort(key=lambda x: x[2], reverse=True)
        top_pairs = scores[:top_n]
        n_pairs = len(top_pairs)

        # Clamp heatmaps_for_top_n to available pairs
        if heatmaps_for_top_n > n_pairs:
            logging.warning(f"Requested {heatmaps_for_top_n} heatmaps but only {n_pairs} pairs available. Showing all.")
            heatmaps_for_top_n = n_pairs
        if heatmaps_for_top_n < 0:
            heatmaps_for_top_n = 0

        # Bar chart of top pairs
        fig, ax = plt.subplots(figsize=(10, 6))
        pair_labels = [f"{p[0]} & {p[1]}" for p in top_pairs]
        mi_vals = [p[2] for p in top_pairs]
        sns.barplot(x=mi_vals, y=pair_labels, palette='Blues_d', ax=ax)
        ax.set_title(f"Top {top_n} Categorical Feature Pairs by Mutual Information with Target")
        ax.set_xlabel('Mutual Information')
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

        # Generate heatmaps for the requested number of top pairs
        for idx, (col1, col2, mi) in enumerate(top_pairs[:heatmaps_for_top_n]):
            logging.info(f"Heatmap {idx+1}: {col1} & {col2} (MI={mi:.4f})")
            self.report.append(f"Categorical interaction heatmap {idx+1}: {col1} & {col2} (MI={mi:.4f})")
            
            combo = df[col1].astype(str) + ' | ' + df[col2].astype(str)
            target_dist = pd.crosstab(combo, df[self.target], normalize='index')
            max_prop = target_dist.max(axis=1)

            # Split only on the first separator to avoid extra columns
            index_df = target_dist.index.to_series().str.split(' | ', n=1, expand=True)
            index_df.columns = [col1, col2]

            heatmap_df = pd.DataFrame({
                col1: index_df[col1],
                col2: index_df[col2],
                'max_prop': max_prop.values
            })
            pivot = heatmap_df.pivot(index=col1, columns=col2, values='max_prop')

            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', linewidths=0.5)
            plt.title(f"Proportion of Majority Class in Combinations of\n{col1} (rows) and {col2} (columns)\n(MI = {mi:.4f})")
            plt.tight_layout()
            plt.show()

        if self.show_guidance:
            self._print_guidance('categorical_interactions')
        plt.close('all')
        
    # -------------------------------------------------------------------------
    # 22. Feature combination predictiveness
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
        df = self._get_df()
        X_base = df[self.num_cols + self.cat_cols].copy()
        if self.cat_cols:
            X_base, _ = self._ordinal_encode_df(X_base, self.cat_cols)
        X_base = X_base.fillna(X_base.median())
        y = df[self.target]
        X_train, X_test, y_train, y_test = train_test_split(X_base, y, test_size=test_size, random_state=42, stratify=y)
        base_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        base_model.fit(X_train, y_train)
        base_acc = accuracy_score(y_test, base_model.predict(X_test))
        hasher = FeatureHasher(n_features=100, input_type='string')
        all_cats_combo = df[self.cat_cols].astype(str).agg('_'.join, axis=1)
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
    # Additional analyses
    # -------------------------------------------------------------------------
    def data_summary(self):
        """Print/log a compact data profile summary."""
        df = self._get_df()
        summary = {
            'shape': df.shape,
            'dtypes': df.dtypes.astype(str).to_dict(),
            'memory_bytes': int(df.memory_usage(deep=True).sum()),
            'missing_counts': df.isnull().sum().to_dict(),
            'missing_pct': (df.isnull().mean() * 100).round(2).to_dict(),
            'duplicate_rows': int(df.duplicated().sum()),
            'constant_columns': [c for c in df.columns if df[c].nunique(dropna=False) <= 1],
            'unique_counts': df.nunique(dropna=False).to_dict()
        }
        logging.info(f"Data summary: shape={summary['shape']}, memory={summary['memory_bytes']} bytes")
        self.report.append(f"Data summary: {summary}")
        if self.show_guidance:
            self._print_guidance('data_summary')
        plt.close('all')
        return summary

    def correlation_heatmap(self):
        """Plot correlation heatmap for numeric features (include target for regression)."""
        df = self._get_df()
        if not self.num_cols:
            logging.info("No numeric columns to compute correlations.")
            return
        if self.task == 'regression' and self.target in df.columns:
            corr_df = df[self.num_cols + [self.target]].corr()
        else:
            corr_df = df[self.num_cols].corr()
        mask = np.triu(np.ones_like(corr_df, dtype=bool))
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_df, mask=mask, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Heatmap')
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        self.report.append('Correlation heatmap generated.')
        if self.show_guidance:
            self._print_guidance('correlation_heatmap')
        plt.close('all')

    def time_series_analysis(self):
        """Simple time-series checks and plots for the first datetime column."""
        df = self._get_df()
        if not self.dt_cols:
            logging.info("No datetime columns available for time-series analysis.")
            return
        time_col = self.time_col or self.dt_cols[0]
        temp = df.copy()
        temp[time_col] = pd.to_datetime(temp[time_col], errors='coerce')
        temp = temp.dropna(subset=[time_col])
        if temp.empty:
            logging.info("No parseable dates found for time-series analysis.")
            return
        temp = temp.sort_values(time_col)
        if self.target not in temp.columns:
            logging.info("Target not available for time-series plotting.")
            return
        series = temp.set_index(time_col)[self.target].dropna()
        fig, ax = plt.subplots(figsize=(12, 6))
        series.plot(ax=ax, label='value')
        window = min(30, max(3, len(series)//10))
        series.rolling(window=window).mean().plot(ax=ax, label=f'rolling_mean_{window}')
        ax.set_title(f'{self.target} over time ({time_col})')
        ax.legend()
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)
        # autocorrelation if available
        try:
            from statsmodels.graphics.tsaplots import plot_acf
            fig2 = plt.figure(figsize=(8, 4))
            plot_acf(series.dropna(), ax=fig2.add_subplot(111), lags=40)
            if self.show_plots:
                plt.show()
            else:
                plt.close(fig2)
        except Exception:
            try:
                from pandas.plotting import autocorrelation_plot
                fig3 = plt.figure(figsize=(8, 4))
                autocorrelation_plot(series.dropna())
                if self.show_plots:
                    plt.show()
                else:
                    plt.close(fig3)
            except Exception:
                logging.info('Autocorrelation plotting not available.')
        if self.show_guidance:
            self._print_guidance('time_series_analysis')
        plt.close('all')

    def data_quality_checks(self):
        """Run lightweight data quality heuristics."""
        df = self._get_df()
        issues = []
        heuristics = ['age', 'year', 'height', 'weight', 'count', 'price', 'quantity']
        for col in df.columns:
            if any(h in col.lower() for h in heuristics) and pd.api.types.is_numeric_dtype(df[col]):
                if (df[col] < 0).any():
                    msg = f"Column '{col}' has negative values but name suggests non-negative."
                    logging.info(msg)
                    issues.append(msg)
            if pd.api.types.is_object_dtype(df[col]):
                vals = df[col].dropna().astype(str)
                uniq = set(vals.str.strip().str.lower().unique())
                na_variants = {'', 'na', 'n/a', 'none', 'null', 'nan'}
                if uniq and uniq.issubset(na_variants):
                    msg = f"Column '{col}' contains only NA-like strings."
                    logging.info(msg)
                    issues.append(msg)
        const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]
        if const_cols:
            msg = f"Constant columns: {const_cols}"
            logging.info(msg)
            issues.append(msg)
        self.report.append(f"Data quality issues: {issues}")
        if self.show_guidance:
            self._print_guidance('data_quality_checks')
        plt.close('all')
        return issues

    def save_report_json(self, filename=None):
        data = self.get_report_dict()
        filename = filename or (self.report_filename.rsplit('.', 1)[0] + '.json')
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=str, indent=2)
        logging.info(f"Report exported to JSON: {filename}")

    def get_report_dict(self):
        """Return a structured report dictionary for programmatic use."""
        return {
            'file_path': self.file_path,
            'shape': self.df_original.shape,
            'target': self.target,
            'task': self.task,
            'num_cols': self.num_cols,
            'cat_cols': self.cat_cols,
            'dt_cols': self.dt_cols,
            'report': self.report
        }

    # -------------------------------------------------------------------------
    # 23. Save final report
    # -------------------------------------------------------------------------
    def save_report(self):
        """Write the accumulated report to a text file with UTF-8 encoding."""
        filename = self.report_filename
        if self.append_timestamp:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, 'txt')
            filename = f"{name}_{ts}.{ext}"
        with open(filename, "w", encoding='utf-8') as f:
            f.write("EDA MASTER REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Dataset: {self.file_path}\n")
            f.write(f"Shape: {self.df_original.shape}\n")
            f.write(f"Target: {self.target}\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Categorical variance threshold: {self.cat_var_threshold}\n\n")
            f.write("-- Summary --\n")
            for line in self.report:
                f.write(str(line) + "\n")
        logging.info(f"✅ Full EDA completed. Report saved as '{filename}'")

    # -------------------------------------------------------------------------
    # 24. Interpretation guidance printer
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
            ,
            'data_summary': (
                "\n[Interpretation] Data summary:\n"
                "  - Check shape, memory, dtypes, missing% and duplicates.\n"
                "  - Constant columns or very high cardinality should be reviewed."
            ),
            'correlation_heatmap': (
                "\n[Interpretation] Correlation heatmap:\n"
                "  - Look for blocks of high correlation (multicollinearity).\n"
                "  - Consider removing or combining highly correlated features."
            ),
            'drift_detection': (
                "\n[Interpretation] Drift detection:\n"
                "  - Low p-values (e.g., <0.05) indicate distributional changes.\n"
                "  - Use drift findings to retrain or adapt models."
            ),
            'encoding_decisions': (
                "\n[Interpretation] Encoding suggestions:\n"
                "  - Low-cardinality: one-hot; high-cardinality: target/frequency encoding.\n"
                "  - For binary features label encoding is simplest."
            ),
            'categorical_interactions': (
                "\n[Interpretation] Categorical interactions:\n"
                "  - High MI pairs may indicate combinatorial features worth creating.\n"
                "  - Heatmaps show which combinations strongly predict a class."
            ),
            'feature_combination_predictiveness': (
                "\n[Interpretation] Feature combination test:\n"
                "  - Compare baseline vs combination model accuracy to gauge utility.\n"
                "  - Small improvements may not justify added complexity."
            ),
            'data_quality_checks': (
                "\n[Interpretation] Data quality checks:\n"
                "  - Watch for negative values in semantically positive columns.\n"
                "  - Whitespace-only or 'N/A' strings should be normalized."
            ),
            'time_series_analysis': (
                "\n[Interpretation] Time-series analysis:\n"
                "  - Look for trends, seasonality, and change points.\n"
                "  - Rolling means and ACF/PACF help identify temporal patterns."
            ),
            'feature_interactions': (
                "\n[Interpretation] Numeric feature interactions:\n"
                "  - Interaction MI highlights non-linear combined effects.\n"
                "  - Consider adding top interactions as features and validating improvements."
            ),
            'missing_as_signal': (
                "\n[Interpretation] Missing-as-signal:\n"
                "  - Missingness correlated with target can be predictive.\n"
                "  - Use missing indicators rather than imputing if informative."
            )
        }
        print(guidance.get(analysis_name, f"\n[Interpretation] No specific guidance for {analysis_name}."))

    # -------------------------------------------------------------------------
    # 25. Orchestration methods
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
    task = input("Task type no | regression: 1 | binary classification: 2 | multi classification: 3: ")

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