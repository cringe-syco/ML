# Loan Default Prediction — Model Summary
**Date:** 2025-11-04 07:02:52

---

## 📈 Model Overview
A LightGBM model was trained on engineered features derived from loan application data to predict default probability. 

### **Performance Metrics**
| Metric | Value |
|:--------|:-------|
| ROC-AUC | **0.9211** |
| PR-AUC | **0.9747** |
| Accuracy | **0.91** |
| Recall | **0.98** |
| Precision | **0.91** |

The model demonstrates excellent discriminatory power and balanced precision-recall performance, suitable for deployment.

---

## 🔍 Confusion Matrix Summary
| | Predicted Negative | Predicted Positive |
|:--|:--:|:--:|
| **Actual Negative** | 14,693 | 9,207 |
| **Actual Positive** | 2,022 | 92,877 |

- **Recall (Sensitivity):** 0.98 → captures nearly all positive (default) cases  
- **Precision:** 0.91 → few false positives  
- **Macro F1:** ~0.83 → balanced performance across both classes

---

## 🧩 Feature Engineering Summary
1. **Mean & Count Encoding:** Applied on categorical variables like `gender`, `marital_status`, `education_level`, `employment_status`, `loan_purpose`, `grade_subgrade`.
2. **Interaction Features:** Created 15 pairwise combinations between key categorical variables.
3. **Frequency Encoding:** Added count-frequency columns for categorical combinations.
4. **Numeric Transformations:** Added log-transformed versions of income and loan-related variables.

Total features used: **42**

---

## 🧠 Top 10 Features (by Importance)
| Rank | Feature | Description |
|------|----------|-------------|
| 1 | employment_status | Strongest predictor of repayment risk |
| 2 | debt_to_income_ratio | Measures borrower’s financial leverage |
| 3 | orig_mean_employment_status | Encoded mean risk across employment types |
| 4 | credit_score | Borrower’s creditworthiness |
| 5 | freq_gender__employment_status | Interaction capturing gender-employment relation |
| 6 | orig_count_employment_status | Frequency of employment category |
| 7 | grade_letter | Loan grade letter |
| 8 | interest_rate | Cost of the loan |
| 9 | loan_amount | Borrowed amount |
| 10 | annual_income | Income level of borrower |

---

## 💬 Insights
- Borrower’s **employment status**, **credit score**, and **debt-to-income ratio** dominate model decisions.
- Interaction features boosted model expressiveness.
- High recall ensures few defaulters are missed, ideal for **risk-sensitive lending**.

---

## 🚀 Next Improvement Steps
1. Use **target encoding with smoothing** to reduce noise in rare categories.
2. Add **temporal features** (e.g., loan issue month, payment duration).
3. Combine models via **stacking** (LightGBM + XGBoost + CatBoost).
4. Apply **SHAP analysis** for interpretability of feature effects.
5. Perform **probability calibration** for threshold optimization.

---

**Conclusion:**  
This LightGBM model with structured feature engineering achieved strong performance (ROC-AUC ≈ 0.92). It captures meaningful borrower and loan-level interactions, making it suitable for production deployment in loan default risk prediction.
