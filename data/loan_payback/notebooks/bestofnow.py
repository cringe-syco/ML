# With Encodings Setup scale pos 1/2.5
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# For preprocessing function

main = pd.read_csv(r"/kaggle/input/playground-series-s5e11/train.csv")
grade_order = main['grade_subgrade'].sort_values().unique().tolist()
grade_subgrade_map = {grade: i+1 for i, grade in enumerate(grade_order)}
grade_subgrade_map

numeric_cols = [
    'log_income_loan_dti', 'grade_num', 
    'annual_income', 'debt_to_income_ratio',
    'interest_rate', 'credit_score',
    'grade_dti_loan', 'grade_num_dti',
    'loan_amount'
]

category_cols = [
    'gender', 'marital_status',
    'education_level', 'employment_status',
    'loan_purpose', 'grade_subgrade'      
]

def preprocessing_new_findings(data):
    data = data.copy()

    # Map grade_subgrade to numeric
    data['grade_num'] = data.grade_subgrade.map(grade_subgrade_map)
    # data = data.drop(columns="grade_subgrade")
    # Grade DTI interaction.
    data['grade_num_dti'] = data.grade_num * data.debt_to_income_ratio

    data['grade_dti_loan'] =  data.grade_num * data.debt_to_income_ratio * np.log10(data.loan_amount)

    data = pd.get_dummies(data=data, columns=category_cols, drop_first=True)

    # Create combined metric feature
    data["log_income_loan_dti"] = (
        np.log10(data['annual_income']) *
        np.log10(data['loan_amount']) *
        data["debt_to_income_ratio"]
    )
    # data[category_cols] = data[category_cols].astype('category')
    return data


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

skf = StratifiedKFold(random_state=42, shuffle=True, n_splits=5)

train_org = pd.read_csv(r"/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv(r"/kaggle/input/playground-series-s5e11/test.csv")

processed_df = preprocessing_new_findings(data=train_org.copy())
processed_test = preprocessing_new_findings(test)

# 🧩 Step 1: Split data
X = processed_df.drop('loan_paid_back', axis=1)  # replace with your target column
X = X.drop('id', axis=1)  # replace with your target column
y = processed_df['loan_paid_back']

X_res = processed_test.drop('id', axis=1)  # Assuming test set doesn't have target
# X_res[test_cols] = X[test_cols].astype('float64')
X_id = processed_test['id']

print(X_res.columns)
print(X.columns)

oof_preds_lgb = np.zeros(len(X))
# oof_preds_xgb = np.zeros(len(X))
test_preds = np.zeros(len(test))



# X[all_cols] = X[all_cols].astype('float64')
# X[cat_cols] = X[cat_cols].astype('category')

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    print(f'--- Fold {fold}/{5} ---')

    X_train, X_test = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[val_idx]

    # ----------------------------------------------------------
    # ⚡ Option 1: LightGBM
    # ----------------------------------------------------------
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        n_estimators=10000,
        learning_rate=0.17898794163735265,
        max_depth=2,
        num_leaves=31,
        reg_lambda=1.2958350559263474,
        reg_alpha=1.0495128632644757,
        subsample=0.5917022549267169,
        min_child_samples=44,
        colsample_bytree=0.5825453457757226,
        random_state=42,
        scale_pos_weight=1.2280728504951048,
        # is_unbalance=True,
        n_jobs=-1,
        early_stopping_rounds=150,
        verbose=-1
    )

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='auc'
    )
    
    y_pred_lgb = lgb_model.predict(X_test)
    y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
    test_preds += lgb_model.predict_proba(X_res, num_iteration=lgb_model.best_iteration_)[:, 1] / 5

    print("\n🔹 LightGBM Performance:")
    print("ROC-AUC:", roc_auc_score(y_test, y_pred_proba_lgb))

    oof_preds_lgb[val_idx] = lgb_model.predict_proba(X_test, num_iteration=lgb_model.best_iteration_)[:, 1]
    print(classification_report(y_test, y_pred_lgb))
    print(confusion_matrix(y_test, (y_pred_lgb > 0.5).astype(int)))


print(f'====================')
print("OOF AUC: LGBM:", roc_auc_score(y, oof_preds_lgb))
print(f'Overall OOF AUC rounded to 4: {roc_auc_score(y, oof_preds_lgb):.4f}')
# print("OOF AUC: XGB", roc_auc_score(y, oof_preds_xgb))
print(f'====================')


# ----------------------------------------------------------
# 🔍 Feature Importance (example for LightGBM)
# ----------------------------------------------------------
import matplotlib.pyplot as plt

lgb.plot_importance(lgb_model, max_num_features=20)
plt.title("Top 15 Important Features (LightGBM)")
plt.show()

# xgb.plot_importance(xgb_model, max_num_features=15, importance_type='gain')
