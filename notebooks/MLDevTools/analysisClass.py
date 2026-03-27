import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging


class EDAAnalysis:
    def __init__(self, file_path, target, task):
        # Initialize necessary parameters
        self.file_path = file_path
        self.target = target
        self.task = task
        self.report = []
        self.df = self.load_data()
        self.num_cols, self.cat_cols, self.dt_cols = self.get_feature_types()

    def load_data(self):
        # Load dataset based on file format
        if self.file_path.endswith('.csv'):
            df = pd.read_csv(self.file_path)
        elif self.file_path.endswith('.xlsx'):
            df = pd.read_excel(self.file_path)        
        elif self.file_path.endswith('.parquet'):
            df = pd.read_parquet(self.file_path)
        else:
            raise ValueError("Unsupported format")
        
        # Log the basic dataframe info
        logging.info(f"Shape: {df.shape}")
        self.report.append(f"Shape: {df.shape}")
        logging.info(f"Data Info: \n{df.info()}")
        
        return df

    def get_feature_types(self):
        # Get feature types: numerical, categorical, datetime
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = self.df.select_dtypes(include='object').columns.tolist()
        dt_cols = self.df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()

        if self.target in num_cols:
            num_cols.remove(self.target)
        if self.target in cat_cols:
            cat_cols.remove(self.target)

        return num_cols, cat_cols, dt_cols

    def target_analysis(self):
        # Target variable analysis
        logging.info("===== TARGET ANALYSIS =====")
        if self.task == "regression":
            logging.info(self.df[self.target].describe())
            sns.histplot(self.df[self.target], kde=True)
            plt.title("Target Distribution")
            plt.show()
        else:
            vc = self.df[self.target].value_counts()
            logging.info(f"Value Counts: \n{vc}")
            sns.barplot(x=vc.index, y=vc.values)
            plt.title("Target Distribution")
            plt.show()

            imbalance_ratio = vc.max() / vc.min()
            logging.info(f"Imbalance Ratio: {imbalance_ratio}")
            self.report.append(f"Imbalance Ratio: {imbalance_ratio}")

    def univariate_analysis(self):
        # Univariate analysis for numerical and categorical columns
        logging.info("===== UNIVARIATE ANALYSIS =====")
        # Numerical
        for col in self.num_cols:
            logging.info(f"\n{col}")
            logging.info(self.df[col].describe())
            sns.histplot(self.df[col], kde=True)
            plt.title(col)
            plt.show()

            sns.boxplot(x=self.df[col])
            plt.title(f"{col} Outliers")
            plt.show()

            skew = self.df[col].skew()
            logging.info(f"Skew: {skew}")
            self.report.append(f"{col} skew: {skew}")

        # Categorical
        for col in self.cat_cols:
            logging.info(f"\n{col}")
            vc = self.df[col].value_counts()
            logging.info(f"Value Counts: \n{vc}")
            vc.plot(kind='bar')
            plt.title(col)
            plt.show()

    def bivariate_analysis(self):
        # Bivariate analysis: Numerical vs Numerical, Categorical vs Numerical
        logging.info("===== BIVARIATE ANALYSIS =====")
        # Numerical vs Numerical
        if len(self.num_cols) > 1:
            sns.pairplot(self.df[self.num_cols])
            plt.show()

            corr = self.df[self.num_cols].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm')
            plt.title("Correlation Matrix")
            plt.show()

        # Categorical vs Numerical
        for col in self.cat_cols:
            for num in self.num_cols:
                sns.boxplot(x=self.df[col], y=self.df[num])
                plt.title(f"{col} vs {num}")
                plt.xticks(rotation=45)
                plt.show()

        # Categorical vs Categorical
        for i in range(len(self.cat_cols)):
            for j in range(i + 1, len(self.cat_cols)):
                cross = pd.crosstab(self.df[self.cat_cols[i]], self.df[self.cat_cols[j]])
                logging.info(f"\n{self.cat_cols[i]} vs {self.cat_cols[j]}")
                logging.info(cross)

    def target_vs_features(self):
        # Target vs Features analysis
        logging.info("===== TARGET vs FEATURES =====")
        for col in self.num_cols:
            if self.task == "regression":
                sns.scatterplot(x=self.df[col], y=self.df[self.target])
            else:
                sns.boxplot(x=self.df[self.target], y=self.df[col])

            plt.title(f"{col} vs Target")
            plt.show()

        for col in self.cat_cols:
            cross = pd.crosstab(self.df[col], self.df[self.target])
            logging.info(f"\n{col} vs Target")
            logging.info(cross)

            cross.plot(kind='bar', stacked=True)
            plt.title(f"{col} vs Target")
            plt.show()

    def correlation_with_target(self):
        # Feature importance (correlation with target)
        logging.info("===== FEATURE IMPORTANCE (Correlation) =====")
        if self.task == "regression":
            corr = self.df.corr()[self.target].sort_values(ascending=False)
            logging.info(f"Correlation with Target: \n{corr}")
            self.report.append(f"Correlation with Target: \n{corr}")

    def grouped_analysis(self):
        # Grouped analysis based on categorical features
        logging.info("===== GROUPED ANALYSIS =====")
        for col in self.cat_cols:
            grouped = self.df.groupby(col)[self.target].mean()
            logging.info(f"\n{col} → Target Mean")
            logging.info(grouped)

            grouped.plot(kind='bar')
            plt.title(f"{col} vs Target Mean")
            plt.show()

    def outliers_detection(self):
        # Outlier detection using Z-score
        logging.info("===== OUTLIERS =====")
        for col in self.num_cols:
            z = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
            outliers = (z > 3).sum()
            logging.info(f"{col}: {outliers}")

    def save_report(self):
        # Save the report to a text file
        with open("eda_master_report.txt", "w") as f:
            for line in self.report:
                f.write(line + "\n")
        logging.info("\n✅ FULL EDA COMPLETED")

    def execute(self):
        # Main execution function
        self.target_analysis()
        self.univariate_analysis()
        self.bivariate_analysis()
        self.target_vs_features()
        self.correlation_with_target()
        self.grouped_analysis()
        self.outliers_detection()
        self.save_report()


# Usage example
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example usage
    file_path = input("Enter dataset path: ")
    target = input("Enter target column: ")
    task = input("Task type (regression/binary/multiclass): ")

    eda = EDAAnalysis(file_path, target, task)
    eda.execute()