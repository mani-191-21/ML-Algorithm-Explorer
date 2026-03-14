import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer,
  ScatterChart, Scatter, PieChart, Pie, Cell, Legend,
  AreaChart, Area, LineChart, Line, RadarChart, Radar, PolarGrid,
  PolarAngleAxis, PolarRadiusAxis, ComposedChart
} from 'recharts';

const API_URL = 'http://localhost:8000';

const COLORS = ['#4f46e5', '#ec4899', '#0ea5e9', '#10b981', '#f97316', '#a855f7', '#facc15', '#ef4444', '#14b8a6', '#3b82f6', '#f43f5e', '#6366f1'];

const ChartPanel = ({ title, children, wide }) => (
  <div className={`glass-panel chart-panel ${wide ? 'wide' : ''}`}>
    <h4 className="chart-title">{title}</h4>
    {children}
  </div>
);

const Visualizations = ({ modelId, model }) => {
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    axios.get(`${API_URL}/chart-data/${modelId}`)
      .then(res => { setChartData(res.data); setLoading(false); })
      .catch(() => setLoading(false));
  }, [modelId]);

  if (loading) return <div style={{ textAlign: 'center', padding: '3rem', color: 'var(--text-muted)' }}><div className="loader" style={{ margin: '0 auto', width: 40, height: 40 }} /></div>;
  if (!chartData || chartData.error) return <div className="glass-panel" style={{ textAlign: 'center', padding: '3rem' }}>No visualization data available for this model.</div>;

  const histEntries = Object.entries(chartData.histograms || {});
  const scatterEntries = Object.entries(chartData.scatters || {});
  const catEntries = Object.entries(chartData.category_dist || {});
  const barCatEntries = Object.entries(chartData.bar_by_category || {});
  const importanceEntries = Object.entries(chartData.feature_importance || {});
  const metrics = chartData.metrics || {};

  /* ── Helper: build histogram bar data ── */
  const buildHistData = (hist) => {
    if (!hist || !hist.bins || !hist.counts) return [];
    return hist.counts.map((c, i) => ({
      range: `${hist.bins[i]}–${hist.bins[i + 1]}`,
      count: c,
    }));
  };

  /* ── Helper: build scatter data ── */
  const buildScatterData = (sc) => {
    if (!sc || !sc.x || !sc.y) return [];
    return sc.x.map((x, i) => ({ x, y: sc.y[i] }));
  };

  return (
    <div className="tab-content animate-fade-in">
      {/* Model Performance Metrics */}
      <div className="glass-panel" style={{ marginBottom: '1.5rem' }}>
        <h3 className="section-title">📊 Model Performance Summary</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: '1rem' }}>
          <div className="metric-card"><div className="metric-label">R² Score</div><div className="metric-value" style={{ color: '#10b981' }}>{metrics.r2 ?? 'N/A'}</div></div>
          <div className="metric-card"><div className="metric-label">RMSE</div><div className="metric-value" style={{ color: '#f97316' }}>{metrics.rmse ?? 'N/A'}</div></div>
          <div className="metric-card"><div className="metric-label">MAE</div><div className="metric-value" style={{ color: '#0ea5e9' }}>{metrics.mae ?? 'N/A'}</div></div>
          <div className="metric-card"><div className="metric-label">Algorithm</div><div className="metric-value">{model.algorithm}</div></div>
        </div>
      </div>

      <div className="charts-grid">
        {/* ── 1. Target Distribution Histogram ── */}
        {chartData.target_hist && (
          <ChartPanel title={`📈 Target Distribution: ${model.target}`} wide>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={buildHistData(chartData.target_hist)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="range" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fill: '#94a3b8' }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                <Bar dataKey="count" fill="#4f46e5" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        {/* ── 2. Feature Importance Bar Chart ── */}
        {importanceEntries.length > 0 && (
          <ChartPanel title="🏆 Feature Importance Ranking" wide>
            <ResponsiveContainer width="100%" height={Math.max(200, importanceEntries.length * 30)}>
              <BarChart data={importanceEntries.map(([k, v]) => ({ name: k, importance: v }))} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis type="number" tick={{ fill: '#94a3b8' }} />
                <YAxis dataKey="name" type="category" width={180} tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                <Bar dataKey="importance" fill="#ec4899" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        {/* ── 3. Predicted vs Actual Scatter ── */}
        {chartData.pred_vs_actual && (
          <ChartPanel title="🎯 Predicted vs Actual Values">
            <ResponsiveContainer width="100%" height={250}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="x" name="Actual" tick={{ fill: '#94a3b8' }} label={{ value: 'Actual', fill: '#94a3b8', position: 'bottom' }} />
                <YAxis dataKey="y" name="Predicted" tick={{ fill: '#94a3b8' }} label={{ value: 'Predicted', fill: '#94a3b8', angle: -90, position: 'left' }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                <Scatter data={chartData.pred_vs_actual.actual.map((a, i) => ({ x: a, y: chartData.pred_vs_actual.predicted[i] }))} fill="#10b981" fillOpacity={0.6} />
              </ScatterChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        {/* ── 4. Residuals Distribution ── */}
        {chartData.residuals && (
          <ChartPanel title="📉 Residuals Distribution">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={buildHistData(chartData.residuals)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="range" tick={{ fill: '#94a3b8', fontSize: 10 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fill: '#94a3b8' }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                <Bar dataKey="count" fill="#f97316" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        {/* ── 5–9. Scatter plots: top features vs target ── */}
        {scatterEntries.slice(0, 5).map(([fname, sc], i) => (
          <ChartPanel key={fname} title={`🔵 ${fname} vs ${model.target}`}>
            <ResponsiveContainer width="100%" height={250}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="x" name={fname} tick={{ fill: '#94a3b8' }} />
                <YAxis dataKey="y" name={model.target} tick={{ fill: '#94a3b8' }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                <Scatter data={buildScatterData(sc)} fill={COLORS[i % COLORS.length]} fillOpacity={0.6} />
              </ScatterChart>
            </ResponsiveContainer>
          </ChartPanel>
        ))}

        {/* ── 10–13. Feature Distribution Histograms ── */}
        {histEntries.slice(0, 4).map(([fname, hist], i) => (
          <ChartPanel key={fname} title={`📊 ${fname} Distribution`}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={buildHistData(hist)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="range" tick={{ fill: '#94a3b8', fontSize: 9 }} angle={-30} textAnchor="end" height={55} />
                <YAxis tick={{ fill: '#94a3b8' }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                <Bar dataKey="count" fill={COLORS[(i + 3) % COLORS.length]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
        ))}

        {/* ── 14. Pie Charts for Categorical Features ── */}
        {catEntries.map(([fname, cats], i) => (
          <ChartPanel key={fname} title={`🥧 ${fname} Category Distribution`}>
            <ResponsiveContainer width="100%" height={250}>
              <PieChart>
                <Pie data={cats.labels.map((l, j) => ({ name: l, value: cats.values[j] }))} cx="50%" cy="50%" outerRadius={80} label>
                  {cats.labels.map((_, j) => <Cell key={j} fill={COLORS[j % COLORS.length]} />)}
                </Pie>
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          </ChartPanel>
        ))}

        {/* ── 15. Bar Charts — Mean target by category ── */}
        {barCatEntries.map(([fname, bar]) => (
          <ChartPanel key={`bar-${fname}`} title={`📊 Average ${model.target} by ${fname}`}>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={bar.labels.map((l, i) => ({ name: l, value: bar.values[i] }))}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                <YAxis tick={{ fill: '#94a3b8' }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                <Bar dataKey="value" fill="#a855f7" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
        ))}

        {/* ── 16. Cumulative Feature Importance ── */}
        {importanceEntries.length > 0 && (
          <ChartPanel title="📈 Cumulative Feature Importance">
            <ResponsiveContainer width="100%" height={250}>
              <AreaChart data={(() => {
                let cum = 0;
                const total = importanceEntries.reduce((s, [, v]) => s + v, 0);
                return importanceEntries.map(([k, v]) => {
                  cum += v;
                  return { name: k, cumulative: parseFloat((cum / total * 100).toFixed(1)) };
                });
              })()}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 9 }} angle={-30} textAnchor="end" height={60} />
                <YAxis tick={{ fill: '#94a3b8' }} domain={[0, 100]} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                <Area type="monotone" dataKey="cumulative" stroke="#10b981" fill="rgba(16,185,129,0.15)" />
              </AreaChart>
            </ResponsiveContainer>
          </ChartPanel>
        )}

        {/* ── 17. Correlation Heatmap (table-based) ── */}
        {chartData.correlation && (
          <ChartPanel title="🔥 Feature Correlation Heatmap" wide>
            <div style={{ overflowX: 'auto' }}>
              <table className="heatmap-table">
                <thead>
                  <tr>
                    <th></th>
                    {chartData.correlation.labels.map(l => <th key={l} style={{ fontSize: '0.7rem', maxWidth: 80, overflow: 'hidden' }}>{l}</th>)}
                  </tr>
                </thead>
                <tbody>
                  {chartData.correlation.matrix.map((row, i) => (
                    <tr key={i}>
                      <td style={{ fontSize: '0.7rem', fontWeight: 600 }}>{chartData.correlation.labels[i]}</td>
                      {row.map((val, j) => {
                        const v = parseFloat(val);
                        const abs = Math.abs(v);
                        const r = v > 0 ? Math.round(abs * 200) : 0;
                        const b = v < 0 ? Math.round(abs * 200) : 0;
                        return (
                          <td key={j} style={{
                            background: `rgba(${r}, ${Math.round(abs * 50)}, ${b}, ${abs * 0.6 + 0.1})`,
                            fontSize: '0.65rem', textAlign: 'center', padding: '4px',
                            color: abs > 0.5 ? '#fff' : '#94a3b8'
                          }}>
                            {v.toFixed(2)}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </ChartPanel>
        )}

        {/* ── 18. More Feature Histograms ── */}
        {histEntries.slice(4).map(([fname, hist], i) => (
          <ChartPanel key={`hist2-${fname}`} title={`📊 ${fname} Distribution`}>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={buildHistData(hist)}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="range" tick={{ fill: '#94a3b8', fontSize: 9 }} angle={-30} textAnchor="end" height={55} />
                <YAxis tick={{ fill: '#94a3b8' }} />
                <Tooltip contentStyle={{ background: '#1e293b', border: '1px solid rgba(255,255,255,0.1)', borderRadius: 8 }} />
                <Bar dataKey="count" fill={COLORS[(i + 7) % COLORS.length]} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartPanel>
        ))}
      </div>
    </div>
  );
};

export default Visualizations;
