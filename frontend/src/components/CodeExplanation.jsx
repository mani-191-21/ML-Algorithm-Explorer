import React from 'react';

/* Per-algorithm Python code explanations with step-by-step annotations */
const CODE = {
  adaboost: {
    title: "AdaBoost Regressor — Python Implementation",
    steps: [
      { heading: "Step 1: Import Libraries", code: `import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error`, explanation: "We import pandas for data manipulation, numpy for numerical ops, AdaBoostRegressor for the model, LabelEncoder for categorical encoding, StandardScaler for feature normalization, and metrics for evaluation." },
      { heading: "Step 2: Load and Explore Data", code: `df = pd.read_csv("insurance.csv")
print(df.shape)          # (1338, 7)
print(df.describe())     # Statistical summary
print(df.isnull().sum()) # Check missing values`, explanation: "Load the insurance dataset. It has 1338 rows and 7 columns: age, sex, bmi, children, smoker, region, charges. Always check shape, stats, and null values first." },
      { heading: "Step 3: Encode Categorical Variables", code: `label_encoders = {}
for col in ['sex', 'smoker', 'region']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
# 'male'→1, 'female'→0
# 'yes'→1, 'no'→0
# 'southwest'→3, 'southeast'→2, etc.`, explanation: "LabelEncoder converts text categories to numbers. We store each encoder so we can inverse-transform later for interpretability." },
      { heading: "Step 4: Split Features and Target", code: `X = df.drop(columns=['charges'])
y = df['charges']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)`, explanation: "Separate the independent variables (X) from the target (y = charges). Split 80% train, 20% test with a fixed random seed for reproducibility." },
      { heading: "Step 5: Feature Scaling", code: `scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)`, explanation: "StandardScaler normalizes features to mean=0, std=1. fit_transform learns the mean/std from training data — transform applies it to test data (never fit on test data!)." },
      { heading: "Step 6: Train the Model", code: `model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=4),
    n_estimators=100,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_scaled, y_train)`, explanation: "Create AdaBoost with 100 decision stumps (max_depth=4) and learning_rate=0.1. Each iteration reweights samples to focus on harder-to-predict cases." },
      { heading: "Step 7: Evaluate", code: `y_pred = model.predict(X_test_scaled)

print(f"R² Score:  {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:      {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"MAE:       {mean_absolute_error(y_test, y_pred):.2f}")`, explanation: "R² tells us how much variance is explained (1.0 = perfect). RMSE and MAE measure average prediction error in dollar terms." },
      { heading: "Step 8: Save Pipeline", code: `import joblib

pipeline = {
    'model': model,
    'scaler': scaler,
    'label_encoders': label_encoders,
    'feature_names': list(X.columns),
}
joblib.dump(pipeline, 'insurance_charges_full_pipeline.pkl')`, explanation: "Save everything needed for production: the trained model, the scaler (to normalize new inputs), and the label encoders (to convert categorical inputs)." },
    ],
  },

  linear: {
    title: "Linear Regression — Python Implementation",
    steps: [
      { heading: "Step 1: Import Libraries", code: `import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error`, explanation: "Linear Regression is the simplest supervised learning algorithm. We import the model class along with preprocessing and evaluation tools." },
      { heading: "Step 2: Load Data", code: `df = pd.read_csv("house-prices.csv")
print(df.columns)  # Area, Bedrooms, Bathrooms, Floors, YearBuilt, Garage, Price`, explanation: "Load the housing dataset. Each row = one house with features like area, bedrooms, and the target: Price." },
      { heading: "Step 3: Prepare Features", code: `X = df.drop(columns=['Price'])
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)`, explanation: "Drop the target column, split into train/test, and scale features. Scaling ensures each feature contributes equally regardless of its original scale." },
      { heading: "Step 4: Train Model", code: `model = LinearRegression()
model.fit(X_train_sc, y_train)

# View learned coefficients
for name, coef in zip(X.columns, model.coef_):
    print(f"{name}: {coef:.2f}")
print(f"Intercept: {model.intercept_:.2f}")`, explanation: "OLS finds the weights that minimize squared error. Each coefficient represents the change in price per unit change in that feature (after scaling)." },
      { heading: "Step 5: Evaluate & Save", code: `y_pred = model.predict(X_test_sc)
print(f"R²: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

import joblib
joblib.dump({'model': model, 'scaler': scaler}, 'pipeline.pkl')`, explanation: "Evaluate on test set and save the pipeline for deployment." },
    ],
  },

  randomforest: {
    title: "Random Forest Regressor — Python Implementation",
    steps: [
      { heading: "Step 1: Import & Load", code: `import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

df = pd.read_csv("cardekho.csv")
df.drop(columns=['name'], inplace=True)`, explanation: "Load car price data and drop the 'name' column (too many unique values to be useful for regression)." },
      { heading: "Step 2: Encode Categoricals", code: `le_dict = {}
for col in ['fuel', 'seller_type', 'transmission', 'owner']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le`, explanation: "Convert categorical text features to numeric encodings." },
      { heading: "Step 3: Split and Scale", code: `X = df.drop(columns=['selling_price'])
y = df['selling_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)`, explanation: "Standard preprocessing pipeline." },
      { heading: "Step 4: Train with Hyperparameters", code: `model = RandomForestRegressor(
    n_estimators=100,    # Number of trees
    max_depth=8,         # Max tree depth (prevents overfitting)
    min_samples_split=5, # Min samples to split a node
    random_state=42,
    n_jobs=-1            # Use all CPU cores
)
model.fit(X_train_sc, y_train)`, explanation: "100 trees, each limited to depth 8. n_jobs=-1 parallelizes training across all CPU cores." },
      { heading: "Step 5: Feature Importance", code: `importances = model.feature_importances_
for name, imp in sorted(zip(X.columns, importances), key=lambda x: -x[1]):
    print(f"{name}: {imp:.4f}")`, explanation: "Random Forest computes feature importance as the mean decrease in impurity (MSE) across all trees where that feature was used to split." },
    ],
  },

  xgboost: {
    title: "XGBoost Regressor — Python Implementation",
    steps: [
      { heading: "Step 1: Import XGBoost", code: `import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error`, explanation: "XGBoost provides its own Python API. Install via: pip install xgboost" },
      { heading: "Step 2: Prepare Data", code: `X = df.drop(columns=['Listening_Time_minutes'])
y = df['Listening_Time_minutes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)`, explanation: "Standard split. XGBoost handles most feature types natively." },
      { heading: "Step 3: Train with Key Parameters", code: `model = xgb.XGBRegressor(
    n_estimators=500,      # Number of boosting rounds
    max_depth=6,           # Tree depth
    learning_rate=0.05,    # Step size shrinkage
    subsample=0.8,         # Row sampling per tree
    colsample_bytree=0.8,  # Feature sampling per tree
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.0,        # L2 regularization
    random_state=42
)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)],
          verbose=50)`, explanation: "Key XGBoost params: learning_rate (smaller = more rounds needed), subsample/colsample (randomness prevents overfitting), reg_alpha/reg_lambda (regularization)." },
      { heading: "Step 4: Cross-Validation", code: `scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print(f"CV R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} ± {scores.std():.4f}")`, explanation: "5-fold cross-validation gives a more reliable estimate of model performance than a single train/test split." },
    ],
  },

  catboost: {
    title: "CatBoost Regressor — Python Implementation",
    steps: [
      { heading: "Step 1: Import CatBoost", code: `from catboost import CatBoostRegressor, Pool
import pandas as pd`, explanation: "CatBoost natively handles categorical features without manual encoding." },
      { heading: "Step 2: Create CatBoost Pool", code: `cat_features = ['Neighborhood', 'ExterQual', 'KitchenQual', ...]
train_pool = Pool(X_train, y_train, cat_features=cat_features)
test_pool = Pool(X_test, y_test, cat_features=cat_features)`, explanation: "CatBoost's Pool object specifies which columns are categorical. They'll be encoded using ordered target statistics internally." },
      { heading: "Step 3: Train", code: `model = CatBoostRegressor(
    iterations=1000,
    depth=6,
    learning_rate=0.05,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=100
)
model.fit(train_pool, eval_set=test_pool)`, explanation: "iterations = boosting rounds, depth = tree depth, l2_leaf_reg = L2 regularization on leaf values." },
    ],
  },

  bayesian: {
    title: "Bayesian Ridge Regression — Python Implementation",
    steps: [
      { heading: "Step 1: Import", code: `from sklearn.linear_model import BayesianRidge`, explanation: "BayesianRidge automatically determines regularization strength from the data." },
      { heading: "Step 2: Train", code: `model = BayesianRidge()
model.fit(X_train_scaled, y_train)

print(f"Alpha (noise): {model.alpha_:.4f}")
print(f"Lambda (reg):  {model.lambda_:.4f}")`, explanation: "Unlike Ridge where you set alpha manually, BayesianRidge learns both the noise precision (alpha) and regularization (lambda) from data via evidence maximization." },
      { heading: "Step 3: Predict with Uncertainty", code: `y_pred, y_std = model.predict(X_test_scaled, return_std=True)
print(f"Prediction: {y_pred[0]:.2f} ± {y_std[0]:.2f}")`, explanation: "BayesianRidge can output prediction uncertainty (standard deviation), which is invaluable for risk-sensitive applications." },
    ],
  },

  elasticnet: {
    title: "ElasticNet Regression — Python Implementation",
    steps: [
      { heading: "Step 1: Import", code: `from sklearn.linear_model import ElasticNet, ElasticNetCV`, explanation: "ElasticNetCV automatically finds optimal alpha and l1_ratio via cross-validation." },
      { heading: "Step 2: Auto-tune with CV", code: `model_cv = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
    alphas=np.logspace(-4, 1, 50),
    cv=5, random_state=42
)
model_cv.fit(X_train_scaled, y_train)
print(f"Best alpha: {model_cv.alpha_:.6f}")
print(f"Best l1_ratio: {model_cv.l1_ratio_:.2f}")`, explanation: "Tests different combinations of alpha (regularization strength) and l1_ratio (Lasso vs Ridge balance). CV selects the best combo." },
    ],
  },

  lasso: {
    title: "Lasso Regression — Python Implementation",
    steps: [
      { heading: "Step 1: Import & Train", code: `from sklearn.linear_model import Lasso, LassoCV

model = LassoCV(cv=5, random_state=42)
model.fit(X_train_scaled, y_train)

# See which features were selected (non-zero)
for name, coef in zip(X.columns, model.coef_):
    if coef != 0:
        print(f"✓ {name}: {coef:.4f}")
    else:
        print(f"✗ {name}: REMOVED")`, explanation: "Lasso's key feature: coefficients exactly equal to zero = automatic feature selection. LassoCV finds optimal alpha." },
    ],
  },

  polynomial: {
    title: "Polynomial Regression — Python Implementation",
    steps: [
      { heading: "Step 1: Feature Expansion", code: `from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline.fit(X_train, y_train)

print(f"Original features: {X_train.shape[1]}")
print(f"Polynomial features: {pipeline['poly'].n_output_features_}")`, explanation: "PolynomialFeatures creates interaction terms and powers. A pipeline chains the transformation + scaling + model together." },
    ],
  },

  ridge: {
    title: "Ridge Regression — Python Implementation",
    steps: [
      { heading: "Step 1: Import & Train", code: `from sklearn.linear_model import Ridge, RidgeCV

# Auto-find best alpha
model = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
model.fit(X_train_scaled, y_train)

print(f"Best alpha: {model.alpha_:.4f}")
print(f"Coefficients: {model.coef_}")`, explanation: "RidgeCV tests 100 alpha values and picks the one with lowest CV error. Unlike Lasso, all coefficients stay non-zero." },
    ],
  },

  svr: {
    title: "Support Vector Regression — Python Implementation",
    steps: [
      { heading: "Step 1: Import & CRITICAL: Scale Features", code: `from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# SVR is VERY sensitive to feature scaling!
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)`, explanation: "CRITICAL: SVR absolutely requires feature scaling. Without it, features with large ranges dominate the kernel computation." },
      { heading: "Step 2: Train with RBF Kernel", code: `model = SVR(
    kernel='rbf',   # Radial Basis Function
    C=100,          # Regularization (higher = fit training data more)
    epsilon=0.1,    # Width of insensitive tube
    gamma='scale'   # Kernel coefficient = 1/(n_features * X.var())
)
model.fit(X_train_sc, y_train)`, explanation: "C controls the trade-off: large C = small margin but fits training data closely. epsilon defines how far a prediction can be from actual without penalty." },
      { heading: "Step 3: Check Support Vectors", code: `print(f"Support vectors: {model.n_support_} out of {len(X_train)}")
print(f"% used: {sum(model.n_support_)/len(X_train)*100:.1f}%")`, explanation: "Only a fraction of training data becomes support vectors. This makes SVR memory-efficient for prediction." },
    ],
  },
};

const CodeExplanation = ({ modelId }) => {
  const codeData = CODE[modelId];
  if (!codeData) return <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>Code explanation not available.</div>;

  return (
    <div className="tab-content animate-fade-in">
      <div className="glass-panel">
        <h2 className="section-title">💻 {codeData.title}</h2>
        <div className="code-steps">
          {codeData.steps.map((step, i) => (
            <div key={i} className="code-step">
              <div className="code-step-header">
                <div className="code-step-num">{i + 1}</div>
                <h3>{step.heading}</h3>
              </div>
              <pre className="code-block"><code>{step.code}</code></pre>
              <div className="code-explanation">
                <strong>💡 Explanation:</strong> {step.explanation}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default CodeExplanation;
