import React from 'react';
import { useNavigate } from 'react-router-dom';
import { ChevronRight, Activity } from 'lucide-react';

const Dashboard = ({ models, iconMap, colorMap }) => {
  const navigate = useNavigate();

  return (
    <div className="animate-fade-in">
      <header className="header">
        <h1 className="page-title">Supervised Learning Hub</h1>
        <p className="page-subtitle">
          Explore <strong>{Object.keys(models).length}</strong> state-of-the-art predictive models.
          Click any model to predict, visualize, and analyze.
        </p>
      </header>
      
      <div className="models-grid">
        {Object.entries(models).map(([id, model]) => {
          const Icon = iconMap[id] || Activity;
          const color = colorMap[id] || 'adaboost-icon';
          
          return (
            <div 
              key={id} 
              className="glass-panel model-card"
              onClick={() => navigate(`/model/${id}`)}
            >
              <div className={`card-icon-wrapper ${color}`}>
                <Icon size={28} />
              </div>
              
              <h3 className="card-title">{model.name}</h3>
              
              <p className="card-desc">
                <span style={{ fontWeight: 500, color: 'var(--text-main)' }}>{model.dataset}</span>
                <br />
                Predicts: <em>{model.target}</em>
              </p>
              
              <div className="card-footer">
                <span className="algo-badge">{model.algorithm}</span>
                <div className="try-now-btn">
                  Explore <ChevronRight size={16} />
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default Dashboard;
