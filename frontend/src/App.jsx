import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { BrainCircuit } from 'lucide-react';
import Dashboard from './components/Dashboard';
import ModelPage from './components/ModelPage';

const API_URL = 'http://localhost:8000';

// Icon mapping
import { Shield, Home, Headphones, Zap, Activity, BarChart3, TrendingUp, Wrench, Car, Bitcoin, LineChart } from 'lucide-react';
const ICON_MAP = {
  adaboost: Shield, catboost: Home, xgboost: Headphones,
  bayesian: Zap, elasticnet: Activity, lasso: BarChart3,
  linear: TrendingUp, polynomial: Wrench, randomforest: Car,
  ridge: Bitcoin, svr: LineChart,
};
const COLOR_MAP = {
  adaboost: 'adaboost-icon', catboost: 'catboost-icon', xgboost: 'xgboost-icon',
  bayesian: 'bayesian-icon', elasticnet: 'elasticnet-icon', lasso: 'lasso-icon',
  linear: 'linear-icon', polynomial: 'polynomial-icon', randomforest: 'randomforest-icon',
  ridge: 'ridge-icon', svr: 'svr-icon',
};

function App() {
  const [models, setModels] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_URL}/models`)
      .then(res => res.json())
      .then(data => { setModels(data); setLoading(false); })
      .catch(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100vh', gap: '1rem', color: 'var(--text-muted)' }}>
        <div className="loader" style={{ width: 40, height: 40 }} />
        <span>Loading MLVerse...</span>
      </div>
    );
  }

  return (
    <Router>
      <div className="app-wrapper">
        {/* Header Bar */}
        <header className="top-bar">
          <div className="top-bar-brand">
            <BrainCircuit size={28} style={{ color: 'var(--primary)' }} />
            <span className="logo-text">MLVerse</span>
          </div>
          <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Supervised Learning Hub · {Object.keys(models).length} Algorithms
          </span>
        </header>

        {/* Main Content — no sidebar */}
        <main className="page-content">
          <Routes>
            <Route path="/" element={<Dashboard models={models} iconMap={ICON_MAP} colorMap={COLOR_MAP} />} />
            <Route path="/model/:modelId" element={<ModelPage models={models} iconMap={ICON_MAP} colorMap={COLOR_MAP} />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
