"""Utility functions to generate Plotly figures for the RiskEngine Flask API."""
import plotly.graph_objects as go

# Colours (same as original Dash app)
BG = '#0d0f12'
SURF = '#151820'
BDR = '#252a35'
TEXT = '#d4d9e8'
MUTED = '#5a6275'
BLUE = '#4a9eff'
PURP = '#9b7ff4'
GREEN = '#2ecc71'
AMBER = '#f0a500'
RED = '#e74c3c'
TEAL = '#1abc9c'

PP = dict(paper_bgcolor=SURF, plot_bgcolor=SURF,
          font=dict(color=TEXT, family='IBM Plex Mono, monospace', size=11))

def ax(t='', g=True):
    return dict(title=t, color=MUTED, showgrid=g, gridcolor=BDR,
                zeroline=False, tickfont=dict(size=10, color=MUTED))

def fig_loss_distribution(mc, var_es):
    """Loss distribution histogram with VaR/ES lines."""
    pnl = mc['pnl']
    v99 = var_es[0.99]['var_montecarlo']
    v95 = var_es[0.95]['var_montecarlo']
    es99 = var_es[0.99]['es_montecarlo']
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pnl/1e5, nbinsx=120, name='P&L distribution',
        marker_color=BLUE, marker_opacity=0.7,
        hovertemplate='P&L: ₹%{x:.1f}L<br>Count: %{y}<extra></extra>'
    ))
    fig.add_vline(x=-v95/1e5, line_color=AMBER, line_width=1.5, line_dash='dash',
                  annotation_text=f'VaR 95%<br>₹{v95/1e5:.1f}L',
                  annotation_font=dict(size=9, color=AMBER), annotation_position='top right')
    fig.add_vline(x=-v99/1e5, line_color=RED, line_width=1.5, line_dash='dash',
                  annotation_text=f'VaR 99%<br>₹{v99/1e5:.1f}L',
                  annotation_font=dict(size=9, color=RED), annotation_position='top right')
    fig.add_vline(x=-es99/1e5, line_color=PURP, line_width=2,
                  annotation_text=f'ES 99%<br>₹{es99/1e5:.1f}L',
                  annotation_font=dict(size=9, color=PURP), annotation_position='top left')
    tail_x = pnl[pnl < -v99]
    if len(tail_x):
        fig.add_trace(go.Histogram(
            x=tail_x/1e5, nbinsx=30, name='Tail (beyond VaR 99%)',
            marker_color=RED, marker_opacity=0.6,
            hovertemplate='Loss: ₹%{x:.1f}L<extra>Tail</extra>'
        ))
    fig.update_layout(**PP, xaxis=ax('Daily P&L (₹ Lakhs)'),
                      yaxis=ax('Frequency'),
                      legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10, color=MUTED)),
                      margin=dict(l=60, r=20, t=20, b=50), barmode='overlay')
    return fig

def fig_var_comparison(var_es):
    """Bar chart comparing VaR and ES across methods."""
    methods = ['Parametric', 'Historical', 'Monte Carlo']
    var_95 = [var_es[0.95][f'var_{m.lower().replace(" ", "_")}'] for m in ['parametric','historical','montecarlo']]
    var_99 = [var_es[0.99][f'var_{m.lower().replace(" ", "_")}'] for m in ['parametric','historical','montecarlo']]
    es_99 = [var_es[0.99][f'es_{m.lower().replace(" ", "_")}'] for m in ['parametric','historical','montecarlo']]
    fig = go.Figure()
    fig.add_trace(go.Bar(name='VaR 95%', x=methods, y=[v/1e5 for v in var_95],
                         marker_color=AMBER, marker_opacity=0.8,
                         hovertemplate='%{x}<br>₹%{y:.2f}L'))
    fig.add_trace(go.Bar(name='VaR 99%', x=methods, y=[v/1e5 for v in var_99],
                         marker_color=RED, marker_opacity=0.8,
                         hovertemplate='%{x}<br>₹%{y:.2f}L'))
    fig.add_trace(go.Bar(name='ES 99% (CVaR)', x=methods, y=[v/1e5 for v in es_99],
                         marker_color=PURP, marker_opacity=0.9,
                         hovertemplate='%{x}<br>₹%{y:.2f}L'))
    fig.update_layout(**PP, barmode='group', xaxis=ax(),
                      yaxis=ax('Risk measure (₹ Lakhs)'),
                      legend=dict(bgcolor='rgba(0,0,0,0)', font=dict(size=10, color=MUTED), orientation='h', y=-0.2),
                      margin=dict(l=60, r=20, t=20, b=80))
    return fig

def fig_copula_comparison(copula, var_es):
    """Bar chart comparing copula VaR against standard methods."""
    labels = ['Historical VaR', 'MC VaR', 'Copula VaR', 'ES (MC)']
    values = [var_es[0.99]['var_historical'], var_es[0.99]['var_montecarlo'], copula.get('copula_var_99'), var_es[0.99]['es_montecarlo']]
    clrs = [BLUE, TEAL, PURP, RED]
    fig = go.Figure(go.Bar(
        x=labels, y=[v/1e5 for v in values],
        marker_color=clrs, marker_opacity=0.85,
        text=[f'₹{v/1e5:.2f}L' for v in values],
        textposition='outside', textfont=dict(size=10,color=MUTED),
        hovertemplate='<b>%{x}</b><br>₹%{y:.2f}L<extra></extra>'
    ))
    fig.update_layout(**PP, xaxis=ax(), yaxis=ax('Risk measure (₹ Lakhs)'),
                      margin=dict(l=50, r=20, t=20, b=60))
    return fig
