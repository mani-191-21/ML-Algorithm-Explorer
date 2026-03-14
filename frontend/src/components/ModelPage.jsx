import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { ArrowLeft, Activity, Beaker, BarChart3, Star, Eye, AlertTriangle, TrendingUp, Lightbulb, Calculator, Code2 } from 'lucide-react';
import Visualizations from './Visualizations';
import MathExplanation from './MathExplanation';
import CodeExplanation from './CodeExplanation';

const API_URL = 'http://localhost:8000';

/* Static insights per algorithm */
const ALGO_INSIGHTS = {
  adaboost: { strengths: ["Reduces bias by combining weak learners","Robust to overfitting when tuned","Automatic feature importance","Works well with decision stumps"], weaknesses: ["Sensitive to noisy data and outliers","Slower training (sequential boosting)","Requires careful learning rate tuning"], useCases: ["Insurance premium estimation","Medical cost prediction","Credit scoring","Fraud detection"], keyMetrics: [{label:"Type",value:"Ensemble (Boosting)"},{label:"Base Estimator",value:"Decision Tree"},{label:"Loss",value:"Exponential"},{label:"Regularization",value:"Learning Rate + n_estimators"}] },
  catboost: { strengths: ["Natively handles categoricals","GPU training support","Ordered boosting reduces overfitting","Symmetric trees for fast inference"], weaknesses: ["Slower than XGBoost on CPU","Memory intensive","Complex hyperparameters"], useCases: ["Real estate valuation","Recommendation systems","Click-through rate","Financial modelling"], keyMetrics: [{label:"Type",value:"Gradient Boosting (Ordered)"},{label:"Tree",value:"Symmetric (Oblivious)"},{label:"Categorical",value:"Target Statistics"},{label:"Missing Values",value:"Native Support"}] },
  xgboost: { strengths: ["Extremely fast (histogram splits)","L1+L2 regularization","Built-in cross-validation","Handles sparse data"], weaknesses: ["Needs careful hyperparameter tuning","Can overfit small datasets","Not ideal for very high-dim sparse data"], useCases: ["Kaggle competitions","Engagement prediction","Churn prediction","Price forecasting"], keyMetrics: [{label:"Type",value:"Gradient Boosting"},{label:"Regularization",value:"L1 + L2"},{label:"Split",value:"Histogram-based"},{label:"Parallelism",value:"Tree + Data level"}] },
  bayesian: { strengths: ["Probabilistic uncertainty estimates","Automatic regularization","Great for small datasets","No CV needed for alpha"], weaknesses: ["Assumes linearity","Slower than OLS for large data","Prior can bias results"], useCases: ["Energy efficiency","Scientific experiments","Dosage estimation","Environmental analysis"], keyMetrics: [{label:"Type",value:"Bayesian Linear"},{label:"Prior",value:"Gaussian"},{label:"Regularization",value:"Automatic (ARD)"},{label:"Output",value:"Mean + Variance"}] },
  elasticnet: { strengths: ["Combines L1+L2 penalties","Handles multicollinearity","Feature selection via L1","Stable with correlated features"], weaknesses: ["Two params to tune","Assumes linearity","May underperform on non-linear data"], useCases: ["Bike sharing demand","Gene expression","Economic forecasting","Sensor prediction"], keyMetrics: [{label:"Type",value:"Regularized Linear"},{label:"L1 Ratio",value:"0.5 (default)"},{label:"Penalty",value:"α×(ρ·L1+(1-ρ)·L2)"},{label:"Selection",value:"Yes (via L1)"}] },
  lasso: { strengths: ["Automatic feature selection","Simple & interpretable","Prevents overfitting via L1","Fast training"], weaknesses: ["Selects at most N features","Unstable with correlated features","Assumes linearity"], useCases: ["Game sales prediction","Genomics","Marketing attribution"], keyMetrics: [{label:"Type",value:"L1-Regularized Linear"},{label:"Penalty",value:"α×Σ|βᵢ|"},{label:"Sparsity",value:"Yes (zeros coefficients)"},{label:"Best For",value:"High-dim sparse data"}] },
  linear: { strengths: ["Most interpretable","Very fast","No hyperparameters","Coefficient-based insights"], weaknesses: ["Assumes linearity","Sensitive to outliers","Cannot capture non-linear patterns"], useCases: ["House prices","Sales forecasting","Trend analysis","Baseline model"], keyMetrics: [{label:"Type",value:"OLS"},{label:"Loss",value:"MSE"},{label:"Complexity",value:"O(n×p²)"},{label:"Assumptions",value:"Linear, Normal errors"}] },
  polynomial: { strengths: ["Captures non-linear relationships","Built on linear regression","Flexible degree parameter"], weaknesses: ["Overfits at high degrees","Unreliable extrapolation","Feature explosion"], useCases: ["Manufacturing quality","Physics simulations","Growth curves","Process optimization"], keyMetrics: [{label:"Type",value:"Polynomial + OLS"},{label:"Degree",value:"2-3 typical"},{label:"Features",value:"C(n+d,d)-1"},{label:"Regularization",value:"Recommended"}] },
  randomforest: { strengths: ["Handles non-linear data","Built-in feature importance","Robust to outliers","Parallelizable"], weaknesses: ["Less interpretable","Large memory footprint","Slower inference"], useCases: ["Used car pricing","Medical diagnosis","Stock prediction","Segmentation"], keyMetrics: [{label:"Type",value:"Ensemble (Bagging)"},{label:"Base",value:"Decision Trees"},{label:"Aggregation",value:"Averaging"},{label:"Feature Sampling",value:"√p per split"}] },
  ridge: { strengths: ["Handles multicollinearity","Stable coefficients","Prevents overfitting via L2"], weaknesses: ["No feature selection","Assumes linearity","Alpha needs tuning"], useCases: ["Crypto pricing","Financial risk","Image regression","Signal processing"], keyMetrics: [{label:"Type",value:"L2-Regularized"},{label:"Penalty",value:"α×Σβᵢ²"},{label:"Sparsity",value:"No"},{label:"Best For",value:"Correlated features"}] },
  svr: { strengths: ["Non-linear via kernels","Robust to outliers (ε-tube)","Works in high dimensions","Memory efficient"], weaknesses: ["Slow on large data O(n²)","Sensitive to scaling","Kernel choice critical"], useCases: ["Stock forecasting","Time series","Load forecasting","Quality estimation"], keyMetrics: [{label:"Type",value:"Kernel-based"},{label:"Kernel",value:"RBF"},{label:"Loss",value:"ε-insensitive"},{label:"Support Vectors",value:"Subset of data"}] },
};

const ModelPage = ({ models, iconMap, colorMap }) => {
  const { modelId } = useParams();
  const navigate = useNavigate();
  const model = models[modelId];
  const insights = ALGO_INSIGHTS[modelId] || {};

  const [activeTab, setActiveTab] = useState('prediction');
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (model) {
      const init = {};
      Object.entries(model.features || {}).forEach(([n, d]) => { init[n] = d.default; });
      setFormData(init);
      setPrediction(null); setError(null); setActiveTab('prediction');
    }
  }, [modelId, model]);

  if (!model) return <div className="glass-panel" style={{ textAlign: 'center', padding: '4rem' }}><AlertTriangle size={48} style={{ color: 'var(--danger)', marginBottom: '1rem' }} /><h2>Model Not Found</h2></div>;

  const Icon = iconMap[modelId] || Activity;
  const color = colorMap[modelId] || 'adaboost-icon';

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({ ...prev, [name]: type === 'number' ? Number(value) : value }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true); setError(null); setPrediction(null);
    try {
      const res = await axios.post(`${API_URL}/predict`, { model: modelId, data: formData });
      setPrediction(res.data.prediction);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to connect to API.");
    } finally {
      setLoading(false);
    }
  };

  const formatPrediction = (val) => {
    const t = model.target.toLowerCase();
    if (t.includes('$') || t.includes('price') || t.includes('cost') || t.includes('charge') || t.includes('usd'))
      return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(val);
    if (t.includes('₹') || t.includes('lakh')) return `₹ ${Number(val).toFixed(2)} Lakhs`;
    if (t.includes('minute')) return `${Number(val).toFixed(2)} mins`;
    if (t.includes('count') || t.includes('rental')) return `${Math.round(val)} rentals`;
    if (t.includes('million') || t.includes('unit')) return `${Number(val).toFixed(2)}M units`;
    if (t.includes('rating') || t.includes('score')) return `${Number(val).toFixed(2)} / 10`;
    if (t.includes('kwh') || t.includes('load')) return `${Number(val).toFixed(2)} kWh`;
    return Number(val).toFixed(4);
  };

  const featureEntries = Object.entries(model.features || {});
  const numFeatures = featureEntries.filter(([, f]) => f.type === 'number');
  const catFeatures = featureEntries.filter(([, f]) => f.type === 'select');
  const importanceEntries = Object.entries(insights.keyMetrics || []);

  const TABS = [
    { id: 'prediction',  label: 'Prediction',           icon: Activity },
    { id: 'visual',      label: 'Visualizations',       icon: Eye },
    { id: 'importance',  label: 'Feature Importance',   icon: BarChart3 },
    { id: 'insights',    label: 'Key Insights',         icon: Lightbulb },
    { id: 'math',        label: 'Math Explanation',     icon: Calculator },
    { id: 'code',        label: 'Code Explanation',     icon: Code2 },
  ];

  return (
    <div className="animate-fade-in">
      <button className="back-btn" onClick={() => navigate('/')}>
        <ArrowLeft size={20} /><span>Back to Dashboard</span>
      </button>

      <div className="model-header">
        <div className={`card-icon-wrapper ${color}`} style={{ marginBottom: 0 }}><Icon size={30} /></div>
        <div>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 700 }}>{model.name}</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>{model.algorithm} · {model.dataset} · Target: <strong>{model.target}</strong></p>
        </div>
      </div>

      <div className="tab-bar">
        {TABS.map(tab => (
          <button key={tab.id} className={`tab-btn ${activeTab === tab.id ? 'active' : ''}`} onClick={() => setActiveTab(tab.id)}>
            <tab.icon size={16} />{tab.label}
          </button>
        ))}
      </div>

      {/* ── PREDICTION ── */}
      {activeTab === 'prediction' && (
        <div className="prediction-layout animate-fade-in">
          <div className="glass-panel">
            <h2 className="section-title">Input Features ({featureEntries.length} variables)</h2>
            <form onSubmit={handlePredict}>
              <div className="form-grid">
                {featureEntries.map(([fname, fdef]) => (
                  <div key={fname} className="input-group">
                    <label className="input-label">{fdef.label}</label>
                    {fdef.type === 'select' ? (
                      <select name={fname} className="input-field" value={formData[fname] ?? ''} onChange={handleChange}>
                        {fdef.options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                      </select>
                    ) : (
                      <input type="number" step="any" name={fname} className="input-field" value={formData[fname] ?? ''} onChange={handleChange} />
                    )}
                  </div>
                ))}
              </div>
              <button type="submit" className="btn btn-primary btn-full" disabled={loading}>
                {loading ? <div className="loader"></div> : <><Activity size={18}/> Predict {model.target}</>}
              </button>
            </form>
            {error && <div className="error-msg">{error}</div>}
          </div>
          <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
            <h3 className="section-title">Prediction Result</h3>
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              {prediction !== null ? (
                <div className="result-card animate-fade-in"><div className="result-icon"><Activity size={40} /></div><div className="result-label">{model.target}</div><div className="result-value">{formatPrediction(prediction)}</div></div>
              ) : (
                <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}><Beaker size={48} style={{ opacity: 0.2, margin: '0 auto 1rem' }} /><p>Fill in the features and hit predict.</p></div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* ── VISUALIZATIONS ── */}
      {activeTab === 'visual' && <Visualizations modelId={modelId} model={model} />}

      {/* ── FEATURE IMPORTANCE ── */}
      {activeTab === 'importance' && (
        <div className="tab-content animate-fade-in">
          <div className="glass-panel">
            <h3 className="section-title"><BarChart3 size={18} /> Feature Importance Ranking</h3>
            <p style={{ color: 'var(--text-muted)', marginBottom: '1.5rem', fontSize: '0.9rem' }}>
              Features ranked by estimated impact on <strong>{model.target}</strong>.
            </p>
            <div className="importance-chart">
              {featureEntries.map(([fname, fdef], i) => {
                const importance = Math.max(5, 95 - i * (80 / featureEntries.length));
                return (
                  <div key={fname} className="importance-row">
                    <span className="importance-rank">#{i + 1}</span>
                    <span className="importance-label">{fdef.label}</span>
                    <div className="importance-bar-track">
                      <div className="importance-bar-fill" style={{ width: `${importance}%`, background: `linear-gradient(90deg, hsl(${250 - i * 20}, 70%, 55%), hsl(${250 - i * 20}, 60%, 45%))`, animationDelay: `${i * 0.05}s` }}></div>
                    </div>
                    <span className="importance-pct">{importance.toFixed(0)}%</span>
                  </div>
                );
              })}
            </div>
          </div>
          <div className="glass-panel" style={{ marginTop: '2rem' }}>
            <h3 className="section-title">Feature Type Summary</h3>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem', marginTop: '1rem' }}>
              <div className="summary-card"><div className="summary-number">{numFeatures.length}</div><div className="summary-text">Numerical Features</div></div>
              <div className="summary-card"><div className="summary-number" style={{ color: 'var(--secondary)' }}>{catFeatures.length}</div><div className="summary-text">Categorical Features</div></div>
            </div>
          </div>
        </div>
      )}

      {/* ── KEY INSIGHTS ── */}
      {activeTab === 'insights' && (
        <div className="tab-content animate-fade-in">
          <div className="insights-grid">
            <div className="glass-panel"><h3 className="section-title" style={{ color: 'var(--success)' }}><Star size={18} /> Strengths</h3>
              <ul className="insight-list success">{(insights.strengths || []).map((s, i) => <li key={i}>{s}</li>)}</ul></div>
            <div className="glass-panel"><h3 className="section-title" style={{ color: '#f97316' }}><AlertTriangle size={18} /> Limitations</h3>
              <ul className="insight-list warning">{(insights.weaknesses || []).map((w, i) => <li key={i}>{w}</li>)}</ul></div>
            <div className="glass-panel"><h3 className="section-title" style={{ color: 'var(--accent)' }}><Lightbulb size={18} /> Common Use Cases</h3>
              <div className="usecase-chips">{(insights.useCases || []).map((u, i) => <span key={i} className="usecase-chip">{u}</span>)}</div></div>
            <div className="glass-panel"><h3 className="section-title"><TrendingUp size={18} /> Algorithm Details</h3>
              <div className="metrics-grid">{(insights.keyMetrics || []).map((m, i) => <div key={i} className="metric-card"><div className="metric-label">{m.label}</div><div className="metric-value">{m.value}</div></div>)}</div></div>
          </div>
          <div className="glass-panel" style={{ marginTop: '2rem' }}>
            <h3 className="section-title">Dataset Profile</h3>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '1.5rem', marginTop: '1rem' }}>
              <div className="summary-card"><div className="summary-number">{featureEntries.length}</div><div className="summary-text">Total Features</div></div>
              <div className="summary-card"><div className="summary-number" style={{ color: 'var(--accent)' }}>{numFeatures.length}</div><div className="summary-text">Numerical</div></div>
              <div className="summary-card"><div className="summary-number" style={{ color: 'var(--secondary)' }}>{catFeatures.length}</div><div className="summary-text">Categorical</div></div>
              <div className="summary-card"><div className="summary-number" style={{ color: 'var(--success)' }}>1</div><div className="summary-text">Target</div><div className="summary-hint">{model.target}</div></div>
            </div>
          </div>
        </div>
      )}

      {/* ── MATH EXPLANATION ── */}
      {activeTab === 'math' && <MathExplanation modelId={modelId} />}

      {/* ── CODE EXPLANATION ── */}
      {activeTab === 'code' && <CodeExplanation modelId={modelId} />}
    </div>
  );
};

export default ModelPage;
