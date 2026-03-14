import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import axios from 'axios';
import { Activity, Beaker, AlertTriangle } from 'lucide-react';

const API_URL = 'http://localhost:8000';

const PredictionForm = ({ models, iconMap, colorMap }) => {
  const { modelId } = useParams();
  const model = models[modelId];
  
  const [formData, setFormData] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [note, setNote] = useState(null);

  useEffect(() => {
    if (model) {
      const initialData = {};
      Object.entries(model.features).forEach(([name, def]) => {
        initialData[name] = def.default;
      });
      setFormData(initialData);
      setPrediction(null);
      setError(null);
      setNote(null);
    }
  }, [modelId, model]);

  if (!model) {
    return (
      <div className="glass-panel" style={{ textAlign: 'center', padding: '4rem' }}>
        <AlertTriangle size={48} style={{ color: 'var(--danger)', marginBottom: '1rem' }} />
        <h2>Model Not Found</h2>
        <p style={{ color: 'var(--text-muted)' }}>The requested model "{modelId}" does not exist.</p>
      </div>
    );
  }

  const Icon = iconMap[modelId] || Activity;
  const color = colorMap[modelId] || 'adaboost-icon';

  const handleChange = (e) => {
    const { name, value, type } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'number' ? Number(value) : value
    }));
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);
    setNote(null);
    
    try {
      const response = await axios.post(`${API_URL}/predict`, {
        model: modelId,
        data: formData
      });
      
      setPrediction(response.data.prediction);
      if (response.data.note) setNote(response.data.note);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to connect to API. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const formatPrediction = (val) => {
    const target = model.target.toLowerCase();
    if (target.includes('$') || target.includes('price') || target.includes('cost') || target.includes('charge') || target.includes('usd')) {
      return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }).format(val);
    }
    if (target.includes('₹') || target.includes('lakh')) {
      return `₹ ${Number(val).toFixed(2)} Lakhs`;
    }
    if (target.includes('minute')) return `${Number(val).toFixed(2)} mins`;
    if (target.includes('count') || target.includes('rental')) return `${Math.round(val)} rentals`;
    if (target.includes('million') || target.includes('unit')) return `${Number(val).toFixed(2)}M units`;
    if (target.includes('rating') || target.includes('score')) return `${Number(val).toFixed(2)} / 10`;
    if (target.includes('kwh') || target.includes('load')) return `${Number(val).toFixed(2)} kWh`;
    return Number(val).toFixed(4);
  };

  return (
    <div className="animate-fade-in">
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', marginBottom: '2rem' }}>
        <div className={`card-icon-wrapper ${color}`} style={{ marginBottom: 0 }}>
          <Icon size={28} />
        </div>
        <div>
          <h1 style={{ fontSize: '1.8rem', fontWeight: 700 }}>{model.name}</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>
            {model.algorithm} · {model.dataset} · Target: <strong>{model.target}</strong>
          </p>
        </div>
      </div>

      <div className="prediction-layout">
        {/* Input Panel */}
        <div className="glass-panel">
          <h2 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '1.5rem', borderBottom: '1px solid var(--panel-border)', paddingBottom: '0.8rem' }}>
            Input Features ({Object.keys(model.features).length} variables)
          </h2>

          <form onSubmit={handlePredict}>
            <div className="form-grid">
              {Object.entries(model.features).map(([fname, fdef]) => (
                <div key={fname} className="input-group">
                  <label className="input-label">{fdef.label}</label>
                  
                  {fdef.type === 'select' ? (
                    <select 
                      name={fname} 
                      className="input-field" 
                      value={formData[fname] ?? ''} 
                      onChange={handleChange}
                    >
                      {fdef.options.map(opt => (
                        <option key={opt} value={opt}>{opt}</option>
                      ))}
                    </select>
                  ) : (
                    <input 
                      type="number" 
                      step="any"
                      name={fname} 
                      className="input-field" 
                      value={formData[fname] ?? ''} 
                      onChange={handleChange}
                    />
                  )}
                </div>
              ))}
            </div>
            
            <button type="submit" className="btn btn-primary btn-full" disabled={loading}>
              {loading ? <div className="loader"></div> : <><Activity size={18}/> Predict {model.target}</>}
            </button>
          </form>
          
          {error && (
            <div style={{ marginTop: '1.5rem', padding: '1rem', background: 'rgba(239,68,68,0.15)', color: '#f87171', border: '1px solid rgba(239,68,68,0.3)', borderRadius: '8px' }}>
              {error}
            </div>
          )}
        </div>

        {/* Output Panel */}
        <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column' }}>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '1rem', borderBottom: '1px solid var(--panel-border)', paddingBottom: '1rem' }}>
            Prediction Result
          </h3>
          
          <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            {prediction !== null ? (
              <div className="result-card animate-fade-in">
                <div className="result-icon">
                  <Activity size={40} />
                </div>
                <div className="result-label">{model.target}</div>
                <div className="result-value">{formatPrediction(prediction)}</div>
                {note && <div className="result-note">{note}</div>}
              </div>
            ) : (
              <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
                <Beaker size={48} style={{ opacity: 0.2, margin: '0 auto 1rem' }} />
                <p>Fill in the features and hit predict to see results.</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionForm;
