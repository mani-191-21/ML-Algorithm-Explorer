import React from 'react';

/* Per-algorithm math explanations with step-by-step formulas */
const MATH = {
  adaboost: {
    title: "AdaBoost Regressor — Mathematical Foundation",
    steps: [
      { heading: "Step 1: Initialize Sample Weights", content: "Assign equal weight to each training sample:", formula: "wᵢ⁽¹⁾ = 1/N  for i = 1, 2, ..., N" },
      { heading: "Step 2: Train Weak Learner (Decision Stump)", content: "Fit a simple decision tree (depth=1) to the weighted training data. The learner minimizes the weighted squared error:", formula: "hₜ(x) = argmin Σᵢ wᵢ⁽ᵗ⁾ · (yᵢ − h(xᵢ))²" },
      { heading: "Step 3: Calculate Weighted Error", content: "Compute the weighted average loss for the t-th learner:", formula: "Lₜ = Σᵢ wᵢ⁽ᵗ⁾ · |yᵢ − hₜ(xᵢ)| / Σᵢ wᵢ⁽ᵗ⁾\n\nL̄ₜ = Lₜ / max(Lₜ)  (normalized to [0,1])" },
      { heading: "Step 4: Compute Learner Weight (Confidence)", content: "The confidence factor βₜ determines how much this learner contributes:", formula: "βₜ = L̄ₜ / (1 − L̄ₜ)\n\nαₜ = ln(1/βₜ)  (higher accuracy → higher weight)" },
      { heading: "Step 5: Update Sample Weights", content: "Increase weights on samples where the learner performed poorly:", formula: "wᵢ⁽ᵗ⁺¹⁾ = wᵢ⁽ᵗ⁾ · βₜ^(1 − |eᵢ|)\n\nwhere eᵢ = |yᵢ − hₜ(xᵢ)| / max|yᵢ − hₜ(xᵢ)|" },
      { heading: "Step 6: Final Prediction (Weighted Median)", content: "Combine all T weak learners using a weighted median:", formula: "F(x) = weighted_median({hₜ(x)}, {αₜ}) for t=1..T\n\nThe learner with the highest R² contributes most." },
      { heading: "Hyperparameters", content: "• n_estimators (T): Number of boosting rounds\n• learning_rate (ν): Shrinkage factor, scales each learner's contribution: αₜ → ν·αₜ\n• base_estimator: Usually DecisionTreeRegressor(max_depth=1-4)" },
    ],
  },

  catboost: {
    title: "CatBoost — Categorical Gradient Boosting",
    steps: [
      { heading: "Step 1: Ordered Target Statistics for Categoricals", content: "Unlike one-hot encoding, CatBoost converts categoricals using ordered target statistics:", formula: "x̂ᵢₖ = (Σⱼ<σ(i) [xⱼₖ = xᵢₖ] · yⱼ + a·P) / (Σⱼ<σ(i) [xⱼₖ = xᵢₖ] + a)\n\nwhere σ is a random permutation, a is smoothing, P is prior." },
      { heading: "Step 2: Initialize Model", content: "Start with the mean of target values:", formula: "F₀(x) = (1/N) Σᵢ yᵢ" },
      { heading: "Step 3: Compute Residuals (Negative Gradient)", content: "For each iteration t, compute the gradient of the loss:", formula: "rᵢ⁽ᵗ⁾ = −∂L(yᵢ, F(xᵢ))/∂F(xᵢ)\n\nFor MSE: rᵢ⁽ᵗ⁾ = yᵢ − Fₜ₋₁(xᵢ)" },
      { heading: "Step 4: Build Symmetric (Oblivious) Decision Tree", content: "CatBoost uses oblivious trees where the same splitting criterion is used across each level:", formula: "hₜ: Splits at each depth d use the same feature+threshold\n→ 2ᵈ leaf nodes with same structure across all paths" },
      { heading: "Step 5: Update Model", content: "Add the new tree with learning rate shrinkage:", formula: "Fₜ(x) = Fₜ₋₁(x) + ν · hₜ(x)\n\nwhere ν is learning_rate ∈ (0, 1]" },
      { heading: "Step 6: Ordered Boosting", content: "To prevent target leakage, CatBoost maintains separate models for different permutations:", formula: "Each sample i gets residuals from a model trained\non only the samples that appeared before i in\na random permutation. This reduces prediction shift." },
      { heading: "Loss Function (RMSE)", formula: "L = √(1/N · Σᵢ (yᵢ − F(xᵢ))²)" },
    ],
  },

  xgboost: {
    title: "XGBoost — Extreme Gradient Boosting",
    steps: [
      { heading: "Step 1: Objective Function", content: "XGBoost minimizes a regularized objective:", formula: "Obj = Σᵢ L(yᵢ, ŷᵢ) + Σₜ Ω(fₜ)\n\nΩ(f) = γT + ½λΣⱼ wⱼ²\nwhere T = #leaves, wⱼ = leaf weights" },
      { heading: "Step 2: Taylor Expansion (Second Order)", content: "Approximate the loss using 2nd order Taylor expansion:", formula: "Obj ≈ Σᵢ [gᵢ·fₜ(xᵢ) + ½hᵢ·fₜ(xᵢ)²] + Ω(fₜ)\n\ngᵢ = ∂L/∂ŷᵢ  (gradient)\nhᵢ = ∂²L/∂ŷᵢ²  (hessian)" },
      { heading: "Step 3: Optimal Leaf Weight", content: "For each leaf j containing instances Iⱼ:", formula: "wⱼ* = −(Σᵢ∈Iⱼ gᵢ) / (Σᵢ∈Iⱼ hᵢ + λ)" },
      { heading: "Step 4: Split Gain Score", content: "For each candidate split, compute the gain:", formula: "Gain = ½ · [GL²/(HL+λ) + GR²/(HR+λ) − (GL+GR)²/(HL+HR+λ)] − γ\n\nGL = Σ(left) gᵢ,  GR = Σ(right) gᵢ" },
      { heading: "Step 5: Tree Building (Histogram)", content: "XGBoost uses histogram-based splits for speed:", formula: "1. Bin continuous features into ~256 buckets\n2. Build gradient histograms per bin\n3. Find best split via scanning cumulative sums\n4. Split → 2 children, recurse until max_depth" },
      { heading: "Step 6: Update Prediction", content: "Add new tree with shrinkage:", formula: "ŷᵢ⁽ᵗ⁾ = ŷᵢ⁽ᵗ⁻¹⁾ + η · fₜ(xᵢ)\n\nη = learning_rate (typically 0.01–0.3)" },
      { heading: "Regularization", content: "L1 (α) and L2 (λ) regularization on leaf weights:", formula: "Ω(f) = γ·T + ½λ·Σwⱼ² + α·Σ|wⱼ|\n\nPrevents overfitting by penalizing complex trees." },
    ],
  },

  bayesian: {
    title: "Bayesian Ridge Regression — Probabilistic Linear Model",
    steps: [
      { heading: "Step 1: Probabilistic Model", content: "Assume a Gaussian likelihood for the data:", formula: "p(y|X, w, α) = N(y | Xw, α⁻¹I)\n\nw = weight vector, α = noise precision" },
      { heading: "Step 2: Prior Distribution on Weights", content: "Place a Gaussian prior on weights:", formula: "p(w|λ) = N(w | 0, λ⁻¹I)\n\nλ = regularization precision (learned, not fixed)" },
      { heading: "Step 3: Posterior Distribution (Bayes' Theorem)", content: "Compute the posterior:", formula: "p(w|y,X,α,λ) = N(w | mN, SN)\n\nmN = α · SN · XᵀY\nSN = (α·XᵀX + λ·I)⁻¹" },
      { heading: "Step 4: Evidence Maximization", content: "Optimize α and λ by maximizing the marginal likelihood:", formula: "log p(y|X,α,λ) = N/2·log(α) + M/2·log(λ)\n  − α/2·||y−Xw||² − λ/2·||w||²\n  − 1/2·log|A| − N/2·log(2π)\nwhere A = α·XᵀX + λ·I" },
      { heading: "Step 5: Prediction with Uncertainty", content: "For a new input x*, predict with uncertainty:", formula: "ŷ* = mNᵀ · x*  (mean prediction)\nσ²* = α⁻¹ + x*ᵀ · SN · x*  (prediction variance)" },
      { heading: "Key Insight", content: "Unlike standard Ridge, Bayesian Ridge automatically finds the optimal regularization strength λ from the data — no cross-validation needed." },
    ],
  },

  elasticnet: {
    title: "ElasticNet Regression — Combined L1 + L2 Penalty",
    steps: [
      { heading: "Step 1: Standard Linear Regression", content: "Without regularization, minimize MSE:", formula: "ŷ = Xw + b\nL = (1/2N) · ||y − Xw||²" },
      { heading: "Step 2: L1 Penalty (Lasso)", content: "Add L1 regularization to enforce sparsity:", formula: "L₁ = α · ρ · Σⱼ |wⱼ|\n\nDrives some weights to exactly zero → feature selection" },
      { heading: "Step 3: L2 Penalty (Ridge)", content: "Add L2 regularization for stability:", formula: "L₂ = α · (1−ρ)/2 · Σⱼ wⱼ²\n\nShrinks weights but keeps all non-zero" },
      { heading: "Step 4: ElasticNet Combined Objective", content: "Combine both penalties:", formula: "L_EN = (1/2N)·||y − Xw||² + α·ρ·Σ|wⱼ| + α·(1−ρ)/2·Σwⱼ²\n\nα = overall regularization strength\nρ = l1_ratio (ρ=1 → Lasso, ρ=0 → Ridge)" },
      { heading: "Step 5: Coordinate Descent Solution", content: "Solved iteratively, one weight at a time:", formula: "wⱼ ← S(rⱼ, α·ρ) / (1 + α·(1−ρ))\n\nS(z, γ) = sign(z) · max(|z| − γ, 0)  (soft threshold)" },
      { heading: "When to Use", content: "• Correlated features (L2 groups them)\n• Want feature selection (L1 zeros some out)\n• p >> n (more features than samples)" },
    ],
  },

  lasso: {
    title: "Lasso Regression — L1 Regularized Linear Model",
    steps: [
      { heading: "Step 1: Objective Function", content: "Minimize MSE with L1 penalty:", formula: "L = (1/2N)·Σᵢ(yᵢ − wᵀxᵢ)² + α·Σⱼ|wⱼ|" },
      { heading: "Step 2: Subgradient Condition", content: "At the optimum, for each weight wⱼ:", formula: "−(1/N)·Σᵢ xᵢⱼ·(yᵢ − ŷᵢ) + α·sign(wⱼ) = 0\n\nIf |partial| < α → wⱼ is set to exactly 0" },
      { heading: "Step 3: Soft Thresholding Operator", content: "The solution uses the soft threshold:", formula: "wⱼ = S(rⱼ, α)\n\nS(z,γ) = { z−γ if z>γ; z+γ if z<−γ; 0 otherwise }" },
      { heading: "Step 4: Coordinate Descent", content: "Optimize one feature at a time:", formula: "For j = 1, 2, ..., p:\n  1. Compute partial residual: rⱼ = (1/N)·XⱼᵀR₋ⱼ\n  2. Update: wⱼ = S(rⱼ, α)\n  3. Update residuals R₋ⱼ\n\nRepeat until convergence." },
      { heading: "Step 5: Feature Selection Property", content: "Lasso's key advantage — automatic feature selection:", formula: "If α → 0: all weights are non-zero (OLS)\nIf α → ∞: all weights are zero\n\nOptimal α found by cross-validation" },
      { heading: "Limitation", content: "Can select at most min(N, p) features. For correlated features, it arbitrarily picks one. ElasticNet addresses this." },
    ],
  },

  linear: {
    title: "Linear Regression — Ordinary Least Squares",
    steps: [
      { heading: "Step 1: Model Definition", content: "Assume linear relationship:", formula: "ŷ = w₁x₁ + w₂x₂ + ... + wₚxₚ + b\n\nIn matrix form: ŷ = Xw + b" },
      { heading: "Step 2: Loss Function (MSE)", content: "Minimize Mean Squared Error:", formula: "L(w) = (1/N) · Σᵢ (yᵢ − wᵀxᵢ − b)²\n\n= (1/N) · ||y − Xw||²" },
      { heading: "Step 3: Gradient", content: "Set derivative to zero:", formula: "∂L/∂w = −(2/N) · Xᵀ(y − Xw) = 0\n\nSolving: XᵀXw = Xᵀy" },
      { heading: "Step 4: Normal Equation (Closed-Form)", content: "Direct solution:", formula: "w* = (XᵀX)⁻¹Xᵀy\n\nComplexity: O(p²·N + p³)" },
      { heading: "Step 5: Interpretation", content: "Each coefficient tells us the marginal effect:", formula: "wⱼ = ∂ŷ/∂xⱼ\n\n\"Holding all else constant, a 1-unit increase\nin xⱼ changes ŷ by wⱼ units.\"" },
      { heading: "Assumptions (Gauss-Markov)", content: "1. Linearity: E[y|X] = Xw\n2. No multicollinearity: XᵀX is invertible\n3. Homoscedasticity: Var(ε) = σ² (constant)\n4. No autocorrelation: Cov(εᵢ, εⱼ) = 0\n5. Normality of errors: ε ~ N(0, σ²)" },
    ],
  },

  polynomial: {
    title: "Polynomial Regression — Non-linear Feature Expansion",
    steps: [
      { heading: "Step 1: Feature Transformation", content: "Expand features to degree d:", formula: "For degree 2, feature x:\n[x] → [x, x²]\n\nFor features [x₁, x₂]:\n→ [x₁, x₂, x₁², x₁x₂, x₂²]" },
      { heading: "Step 2: Number of New Features", content: "Feature count grows combinatorially:", formula: "New features = C(n+d, d) − 1\n\nFor n=5 features, d=2: C(7,2)−1 = 20 features\nFor n=5 features, d=3: C(8,3)−1 = 55 features" },
      { heading: "Step 3: Fit Linear Regression on Expanded Features", content: "After expansion, apply OLS:", formula: "ŷ = w₀ + w₁x₁ + w₂x₂ + w₃x₁² + w₄x₁x₂ + w₅x₂²\n\nSolve via Normal Equation on expanded X'" },
      { heading: "Step 4: Bias-Variance Trade-off", content: "Degree controls model complexity:", formula: "Low degree (d=1): High bias, low variance (underfit)\nHigh degree (d=5+): Low bias, high variance (overfit)\nOptimal: d=2 or 3 (validated by CV)" },
      { heading: "Step 5: Prediction", content: "For new input, first transform then predict:", formula: "1. Transform x_new → [x, x², x³, ...]\n2. ŷ = wᵀ · φ(x_new)\n\nwhere φ(x) is the polynomial feature map" },
      { heading: "Regularization Advice", content: "High-degree polynomials should use Ridge/Lasso to prevent coefficient explosion:\n\nL = ||y − Xw||² + α||w||²" },
    ],
  },

  randomforest: {
    title: "Random Forest Regressor — Ensemble of Decision Trees",
    steps: [
      { heading: "Step 1: Bootstrap Sampling", content: "Create B random subsets of training data with replacement:", formula: "For b = 1, 2, ..., B:\n  Draw D_b of size N from D with replacement\n  ~63.2% unique samples per bootstrap" },
      { heading: "Step 2: Build Decision Trees", content: "For each bootstrap sample, grow a full tree:", formula: "At each node:\n1. Randomly select m features (m = √p for regression)\n2. Find best split among these m features\n3. Split: argmin Σ(left) (yᵢ − ȳ_L)² + Σ(right) (yᵢ − ȳ_R)²" },
      { heading: "Step 3: Leaf Prediction", content: "Each leaf predicts the mean of its samples:", formula: "ŷ_leaf = (1/|S|) · Σᵢ∈S yᵢ\n\nwhere S is the set of training samples in the leaf" },
      { heading: "Step 4: Aggregation (Averaging)", content: "Final prediction = average of all trees:", formula: "ŷ(x) = (1/B) · Σ_{b=1}^B Tᵦ(x)\n\nVariance decreases by factor ~1/B" },
      { heading: "Step 5: Feature Importance (Impurity Decrease)", content: "Computed as mean decrease in MSE across all trees:", formula: "Importance(xⱼ) = (1/B) Σ_b Σ_{node uses xⱼ} Δ(MSE)\n\nΔ(MSE) = MSE_parent − (N_L·MSE_L + N_R·MSE_R)/N" },
      { heading: "Step 6: Out-of-Bag (OOB) Error", content: "Each tree has ~36.8% samples not used in training:", formula: "OOB_error = (1/N) Σᵢ (yᵢ − ŷ_OOB(xᵢ))²\n\nFree validation without a separate test set!" },
    ],
  },

  ridge: {
    title: "Ridge Regression — L2 Regularized Linear Model",
    steps: [
      { heading: "Step 1: Objective Function", content: "Add L2 penalty to OLS:", formula: "L(w) = ||y − Xw||² + α·||w||²\n\n= Σᵢ(yᵢ − wᵀxᵢ)² + α·Σⱼwⱼ²" },
      { heading: "Step 2: Gradient and Solution", content: "Set gradient to zero:", formula: "∂L/∂w = −2Xᵀ(y − Xw) + 2αw = 0\n\nw* = (XᵀX + αI)⁻¹Xᵀy" },
      { heading: "Step 3: Regularization Effect", content: "α controls shrinkage:", formula: "α = 0: Same as OLS\nα → ∞: All weights → 0\n\nOptimal α found via cross-validation\nor Generalized Cross-Validation (GCV)" },
      { heading: "Step 4: SVD Interpretation", content: "Using SVD of X = UDVᵀ:", formula: "w_ridge = V · diag(dⱼ²/(dⱼ²+α)) · Vᵀ · w_OLS\n\nSmall singular values (dⱼ) → heavily shrunk\nLarge singular values → barely changed" },
      { heading: "Step 5: Bias-Variance Trade-off", content: "Ridge adds bias to reduce variance:", formula: "E[||w_ridge − w_true||²] = bias² + variance\n\nbias² ↑ as α ↑ (more shrinkage)\nvariance ↓ as α ↑ (more stable)\n\nOptimal α balances both." },
      { heading: "vs. Lasso", content: "Ridge: shrinks but never zeros coefficients\nLasso: produces sparse solutions (sets weights to 0)\nRidge better for: correlated features, no feature selection needed" },
    ],
  },

  svr: {
    title: "Support Vector Regression — Kernel-Based Regression",
    steps: [
      { heading: "Step 1: ε-Insensitive Loss", content: "SVR ignores errors within an ε tube:", formula: "L_ε(y, ŷ) = max(0, |y − ŷ| − ε)\n\nErrors < ε are considered zero (ε-insensitive)" },
      { heading: "Step 2: Primal Optimization Problem", content: "Minimize model complexity + loss:", formula: "min (1/2)||w||² + C·Σᵢ(ξᵢ + ξᵢ*)\n\nsubject to:\nyᵢ − wᵀφ(xᵢ) − b ≤ ε + ξᵢ\nwᵀφ(xᵢ) + b − yᵢ ≤ ε + ξᵢ*\nξᵢ, ξᵢ* ≥ 0" },
      { heading: "Step 3: Kernel Trick", content: "Use RBF kernel to map to infinite-dimensional space:", formula: "K(xᵢ, xⱼ) = exp(−γ||xᵢ − xⱼ||²)\n\nγ = 1/(2σ²), controls influence radius\nNo need to compute φ(x) explicitly!" },
      { heading: "Step 4: Dual Formulation", content: "Solve the dual problem:", formula: "ŷ(x) = Σᵢ (αᵢ − αᵢ*) · K(xᵢ, x) + b\n\nOnly support vectors (αᵢ ≠ 0) contribute\nTypically a small fraction of training data" },
      { heading: "Step 5: Hyperparameters", content: "Three critical hyperparameters:", formula: "C: Trade-off between model simplicity and tolerance\n  Large C → fit training data closely\n  Small C → wider margin, more regularization\n\nε: Width of insensitive tube\n  Large ε → more samples inside tube\n\nγ: RBF kernel width\n  Large γ → each sample has local influence\n  Small γ → smoother, more global model" },
      { heading: "Key Insight", content: "SVR is memory-efficient (stores only support vectors) and excels in small-to-medium datasets where non-linear relationships exist." },
    ],
  },
};

const MathExplanation = ({ modelId }) => {
  const math = MATH[modelId];
  if (!math) return <div className="glass-panel" style={{ padding: '3rem', textAlign: 'center' }}>Math explanation not available.</div>;

  return (
    <div className="tab-content animate-fade-in">
      <div className="glass-panel">
        <h2 className="section-title">📐 {math.title}</h2>
        <div className="math-steps">
          {math.steps.map((step, i) => (
            <div key={i} className="math-step">
              <div className="math-step-header">
                <div className="math-step-num">{i + 1}</div>
                <h3>{step.heading}</h3>
              </div>
              {step.content && <p className="math-content">{step.content}</p>}
              {step.formula && <pre className="math-formula">{step.formula}</pre>}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default MathExplanation;
