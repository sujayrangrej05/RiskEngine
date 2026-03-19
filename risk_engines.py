"""
risk_engines.py
5 quantitative market risk engines — Basel III compliant

Engine 1: Monte Carlo simulation (GBM + Cholesky decomposition)
Engine 2: VaR + Expected Shortfall (Parametric, Historical, Monte Carlo)
Engine 3: Historical stress testing (GFC 2008, COVID-19, Rate hike 2022)
Engine 4: Gaussian copula tail dependence model
Engine 5: VaR backtesting (Kupiec POF test + Basel traffic light)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, chi2
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════════

def load_data(path='prices.csv'):
    prices = pd.read_csv(path, index_col=0, parse_dates=True)
    returns = prices.pct_change().dropna()
    return prices, returns


# ══════════════════════════════════════════════════════════════════
# ENGINE 1 — MONTE CARLO SIMULATION
# ══════════════════════════════════════════════════════════════════

def monte_carlo_simulation(returns, weights, portfolio_value=10_000_000,
                            n_simulations=10_000, horizon=1, seed=42):
    """
    GBM Monte Carlo with Cholesky-decomposed correlated returns.

    Steps:
    1. Estimate mu (drift) and sigma (volatility) per asset
    2. Compute Cholesky decomposition of correlation matrix
    3. Draw correlated random shocks
    4. Apply GBM formula: dS = mu*dt + sigma*sqrt(dt)*Z
    5. Compute portfolio P&L distribution

    Args:
        returns:          DataFrame of daily returns
        weights:          array of portfolio weights (must sum to 1)
        portfolio_value:  total portfolio in INR
        n_simulations:    number of Monte Carlo paths
        horizon:          forecast horizon in days (1 = 1-day VaR)

    Returns dict with full P&L distribution and simulation paths.
    """
    np.random.seed(seed)
    n_assets = len(weights)
    weights  = np.array(weights)
    dt       = 1 / 252

    # Annualised parameters
    mu    = returns.mean().values * 252
    sigma = returns.std().values  * np.sqrt(252)

    # Correlation structure via Cholesky
    corr_matrix = returns.corr().values
    L = np.linalg.cholesky(corr_matrix)     # lower triangular: L @ L.T = Corr

    # Simulate correlated shocks for each asset over horizon days
    all_pnl = np.zeros(n_simulations)
    paths   = np.zeros((n_simulations, n_assets))

    for sim in range(n_simulations):
        cumulative_return = np.zeros(n_assets)
        for _ in range(horizon):
            z_indep  = np.random.standard_normal(n_assets)
            z_corr   = L @ z_indep          # correlated shocks
            dr = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_corr
            cumulative_return += dr

        asset_pnl    = portfolio_value * weights * cumulative_return
        all_pnl[sim] = asset_pnl.sum()
        paths[sim]   = cumulative_return

    return {
        'pnl':          all_pnl,
        'paths':        paths,
        'n_sims':       n_simulations,
        'horizon':      horizon,
        'portfolio_value': portfolio_value,
        'weights':      weights,
        'assets':       list(returns.columns),
        'mu':           mu,
        'sigma':        sigma,
        'corr':         corr_matrix,
        'cholesky_L':   L,
    }


# ══════════════════════════════════════════════════════════════════
# ENGINE 2 — VAR AND EXPECTED SHORTFALL
# ══════════════════════════════════════════════════════════════════

def calculate_var_es(mc_result, returns, weights, portfolio_value=10_000_000,
                     confidence_levels=(0.95, 0.99)):
    """
    Three VaR methods + Expected Shortfall (CVaR) — Basel III standard.

    Methods:
    - Parametric (Delta-Normal): assumes normal distribution of returns
    - Historical: empirical quantile of observed P&L — no distribution assumption
    - Monte Carlo: quantile of simulated P&L distribution

    Expected Shortfall (ES / CVaR):
        ES = E[Loss | Loss > VaR]
        Basel III moved from VaR to ES at 97.5% as primary measure after 2008.
        ES captures tail risk better than VaR (VaR says nothing about beyond the threshold).
    """
    pnl_mc    = mc_result['pnl']
    weights_a = np.array(weights)
    port_ret  = (returns * weights_a).sum(axis=1)

    results = {}
    for cl in confidence_levels:
        z = norm.ppf(1 - cl)

        # Parametric VaR
        port_mu    = port_ret.mean()
        port_sigma = port_ret.std()
        var_param  = -(port_mu + z * port_sigma) * portfolio_value

        # Historical VaR
        hist_pnl   = port_ret.values * portfolio_value
        var_hist   = -np.percentile(hist_pnl, (1 - cl) * 100)

        # Monte Carlo VaR
        var_mc     = -np.percentile(pnl_mc, (1 - cl) * 100)

        # Expected Shortfall (all three methods)
        # Parametric ES: closed-form for normal distribution
        es_param = -(port_mu - port_sigma * norm.pdf(norm.ppf(1-cl)) / (1-cl)) * portfolio_value

        # Historical ES: mean of losses beyond VaR
        threshold_hist = np.percentile(hist_pnl, (1 - cl) * 100)
        es_hist  = -hist_pnl[hist_pnl <= threshold_hist].mean()

        # Monte Carlo ES
        threshold_mc = np.percentile(pnl_mc, (1 - cl) * 100)
        es_mc    = -pnl_mc[pnl_mc <= threshold_mc].mean()

        results[cl] = {
            'var_parametric':  round(var_param,  2),
            'var_historical':  round(var_hist,   2),
            'var_montecarlo':  round(var_mc,     2),
            'es_parametric':   round(es_param,   2),
            'es_historical':   round(es_hist,    2),
            'es_montecarlo':   round(es_mc,      2),
            # Basel III primary measure (ES at 97.5%)
            'basel_es':        round(es_mc, 2) if abs(cl - 0.975) < 0.01 else None,
        }

    # Component VaR (marginal contribution per asset)
    component_var = {}
    for i, asset in enumerate(returns.columns):
        w_perturbed     = weights_a.copy()
        w_perturbed[i] += 0.001
        w_perturbed    /= w_perturbed.sum()
        port_ret_p      = (returns * w_perturbed).sum(axis=1)
        var_p           = -np.percentile(port_ret_p.values * portfolio_value,
                                          (1 - 0.99) * 100)
        var_base        = results[0.99]['var_historical']
        component_var[asset] = round((var_p - var_base) / 0.001 * weights_a[i], 2)

    results['component_var'] = component_var
    results['portfolio_vol'] = round(port_ret.std() * np.sqrt(252) * 100, 3)
    results['portfolio_ret'] = round(port_ret.mean() * 252 * 100, 3)
    results['sharpe']        = round(
        (port_ret.mean() * 252 - 0.065) / (port_ret.std() * np.sqrt(252)), 3)
    return results


# ══════════════════════════════════════════════════════════════════
# ENGINE 3 — HISTORICAL STRESS TESTING
# ══════════════════════════════════════════════════════════════════

# Real historical drawdowns for major Indian market crises
STRESS_SCENARIOS = {
    'GFC 2008 (Lehman collapse)': {
        'description': 'Global Financial Crisis — Lehman Brothers collapse Sep 2008. Nifty50 fell 60% peak-to-trough.',
        'asset_shocks': {
            'RELIANCE': -0.58, 'TCS': -0.52, 'HDFCBANK': -0.65, 'INFY': -0.50,
            'ICICIBANK': -0.72, 'SBIN': -0.68, 'BHARTI': -0.48, 'ITC': -0.38,
            'KOTAKBANK': -0.62, 'HINDUNILVR': -0.30,
        },
        'duration_days': 180,
        'recovery_days': 540,
        'color': '#e74c3c',
    },
    'COVID-19 crash (Mar 2020)': {
        'description': 'COVID-19 pandemic crash — Nifty50 fell 38% in 40 days. Fastest bear market in history.',
        'asset_shocks': {
            'RELIANCE': -0.38, 'TCS': -0.30, 'HDFCBANK': -0.42, 'INFY': -0.28,
            'ICICIBANK': -0.48, 'SBIN': -0.52, 'BHARTI': -0.32, 'ITC': -0.40,
            'KOTAKBANK': -0.44, 'HINDUNILVR': -0.22,
        },
        'duration_days': 40,
        'recovery_days': 180,
        'color': '#f0a500',
    },
    'Rate hike shock (2022)': {
        'description': 'RBI rate hike cycle + global tightening. Nifty fell 17% from Jan–Jun 2022.',
        'asset_shocks': {
            'RELIANCE': -0.12, 'TCS': -0.22, 'HDFCBANK': -0.15, 'INFY': -0.26,
            'ICICIBANK': -0.14, 'SBIN': -0.10, 'BHARTI': -0.12, 'ITC': -0.08,
            'KOTAKBANK': -0.16, 'HINDUNILVR': -0.14,
        },
        'duration_days': 120,
        'recovery_days': 240,
        'color': '#9b7ff4',
    },
    'Demonetisation shock (Nov 2016)': {
        'description': 'India demonetisation — ₹500/₹1000 notes withdrawn overnight. Nifty fell 6% in 2 days.',
        'asset_shocks': {
            'RELIANCE': -0.06, 'TCS': -0.04, 'HDFCBANK': -0.08, 'INFY': -0.04,
            'ICICIBANK': -0.10, 'SBIN': -0.12, 'BHARTI': -0.05, 'ITC': -0.18,
            'KOTAKBANK': -0.09, 'HINDUNILVR': -0.12,
        },
        'duration_days': 5,
        'recovery_days': 60,
        'color': '#1abc9c',
    },
    'Severe tail risk (-3σ)': {
        'description': 'Hypothetical 3-sigma event — calibrated to worst 0.13% of observed returns.',
        'asset_shocks': {s: -0.22 for s in ['RELIANCE','TCS','HDFCBANK','INFY',
                                              'ICICIBANK','SBIN','BHARTI','ITC',
                                              'KOTAKBANK','HINDUNILVR']},
        'duration_days': 1,
        'recovery_days': 30,
        'color': '#e056b0',
    },
}


def run_stress_tests(weights, portfolio_value=10_000_000):
    """
    Apply historical shock scenarios to portfolio.
    For each scenario: P&L impact = sum(weight_i * shock_i) * portfolio_value
    """
    weights_arr = np.array(weights)
    assets      = ['RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK',
                   'SBIN','BHARTI','ITC','KOTAKBANK','HINDUNILVR']
    results = {}
    for scenario_name, scenario in STRESS_SCENARIOS.items():
        shocks  = np.array([scenario['asset_shocks'].get(a, -0.15) for a in assets])
        pnl     = (weights_arr * shocks * portfolio_value).sum()
        pnl_pct = (weights_arr * shocks).sum() * 100

        # Time to recovery estimate (simple)
        recovery_rate = abs(pnl_pct) / scenario['recovery_days']

        results[scenario_name] = {
            'pnl':           round(pnl,     2),
            'pnl_pct':       round(pnl_pct, 3),
            'duration_days': scenario['duration_days'],
            'recovery_days': scenario['recovery_days'],
            'recovery_rate': round(recovery_rate, 3),
            'description':   scenario['description'],
            'color':         scenario['color'],
            'asset_impacts': {
                a: round(weights_arr[i] * scenario['asset_shocks'].get(a, -0.15) * portfolio_value, 2)
                for i, a in enumerate(assets)
            },
        }
    return results


# ══════════════════════════════════════════════════════════════════
# ENGINE 4 — GAUSSIAN COPULA
# ══════════════════════════════════════════════════════════════════

def gaussian_copula_simulation(returns, weights, portfolio_value=10_000_000,
                                 n_simulations=10_000, seed=42):
    """
    Gaussian copula captures realistic tail dependence between assets.

    Why copulas over simple correlation:
    - Simple Pearson correlation assumes joint normality
    - During crashes, correlations spike — assets that seemed independent move together
    - Copulas separate the marginal distributions from the dependency structure
    - Gaussian copula: use empirical marginals (actual fat tails) + normal dependency

    Steps:
    1. Transform each return series to uniform [0,1] via empirical CDF (rank transform)
    2. Apply inverse normal CDF → pseudo-observations in normal space
    3. Fit correlation matrix to pseudo-observations
    4. Cholesky decompose and simulate correlated normals
    5. Transform back via empirical quantile function → realistic returns
    6. Compute portfolio P&L
    """
    np.random.seed(seed)
    n_assets = returns.shape[1]
    weights_a = np.array(weights)
    n_obs     = len(returns)

    # Step 1-2: Rank transform → pseudo-observations
    pseudo = np.zeros_like(returns.values, dtype=float)
    for i in range(n_assets):
        ranks = stats.rankdata(returns.values[:, i])
        pseudo[:, i] = norm.ppf(ranks / (n_obs + 1))   # Hazen plotting position

    # Step 3: Correlation of pseudo-observations
    corr_pseudo = np.corrcoef(pseudo.T)
    corr_pseudo = np.clip(corr_pseudo, -0.999, 0.999)
    np.fill_diagonal(corr_pseudo, 1.0)

    # Step 4: Cholesky
    try:
        L = np.linalg.cholesky(corr_pseudo)
    except np.linalg.LinAlgError:
        # Regularise if not PD
        corr_pseudo += np.eye(n_assets) * 1e-6
        L = np.linalg.cholesky(corr_pseudo)

    # Step 5-6: Simulate
    all_pnl = np.zeros(n_simulations)
    for sim in range(n_simulations):
        z        = np.random.standard_normal(n_assets)
        z_corr   = L @ z
        u        = norm.cdf(z_corr)       # back to uniform
        u        = np.clip(u, 1e-6, 1-1e-6)
        sim_rets = np.array([
            np.quantile(returns.values[:, i], u[i])
            for i in range(n_assets)
        ])
        all_pnl[sim] = (weights_a * sim_rets * portfolio_value).sum()

    # Compare tail dependence: how much worse is copula VaR vs parametric?
    copula_var_99  = -np.percentile(all_pnl, 1.0)
    simple_var_99  = -np.percentile(
        (returns.values @ weights_a) * portfolio_value, 1.0)
    tail_amplification = (copula_var_99 - simple_var_99) / simple_var_99 * 100

    return {
        'pnl':                all_pnl,
        'copula_var_95':      round(-np.percentile(all_pnl, 5.0), 2),
        'copula_var_99':      round(copula_var_99, 2),
        'copula_es_99':       round(-all_pnl[all_pnl <= np.percentile(all_pnl, 1)].mean(), 2),
        'tail_amplification': round(tail_amplification, 2),
        'corr_pseudo':        corr_pseudo,
        'simple_var_99':      round(simple_var_99, 2),
    }


# ══════════════════════════════════════════════════════════════════
# ENGINE 5 — VAR BACKTESTING (KUPIEC TEST)
# ══════════════════════════════════════════════════════════════════

def backtest_var(returns, weights, portfolio_value=10_000_000,
                 confidence_level=0.99, window=250):
    """
    Basel II/III VaR backtesting — Kupiec Proportion of Failures (POF) test.

    Regulatory requirement: Banks must count how many days actual loss exceeds VaR.
    Basel II traffic light system (over 250 trading days):
        Green zone:  0-4  exceptions → model approved
        Amber zone:  5-9  exceptions → model under review (capital add-on)
        Red zone:   10+   exceptions → model rejected

    Kupiec POF test:
        H0: p (true exception rate) = 1 - confidence level
        Test statistic: LR = -2 ln[(1-p0)^(T-N) * p0^N] + 2 ln[(1-N/T)^(T-N) * (N/T)^N]
        Under H0: LR ~ chi2(1)
    """
    weights_a = np.array(weights)
    port_ret  = (returns * weights_a).sum(axis=1).values
    n_total   = len(port_ret)

    # Rolling VaR (re-estimated every day on trailing window)
    var_series      = []
    actual_pnl      = []
    exception_flags = []
    dates           = returns.index[window:]

    for t in range(window, n_total):
        window_ret = port_ret[t-window:t]
        var_1d     = -np.percentile(window_ret, (1-confidence_level)*100) * portfolio_value
        today_pnl  = port_ret[t] * portfolio_value

        var_series.append(var_1d)
        actual_pnl.append(today_pnl)
        exception_flags.append(today_pnl < -var_1d)

    var_series      = np.array(var_series)
    actual_pnl      = np.array(actual_pnl)
    exception_flags = np.array(exception_flags)

    N = exception_flags.sum()     # number of exceptions
    T = len(exception_flags)      # total observations
    p0 = 1 - confidence_level     # expected exception rate

    # Kupiec LR test statistic
    p_hat = N / T if N > 0 else 1e-10
    LR = -2 * (
        N * np.log(p0 / p_hat) + (T - N) * np.log((1 - p0) / (1 - p_hat))
    ) if 0 < p_hat < 1 else 0
    p_value = 1 - chi2.cdf(LR, df=1)

    # Basel traffic light
    if N <= 4:
        zone, zone_color = 'Green', '#2ecc71'
    elif N <= 9:
        zone, zone_color = 'Amber', '#f0a500'
    else:
        zone, zone_color = 'Red', '#e74c3c'

    return {
        'var_series':        var_series,
        'actual_pnl':        actual_pnl,
        'exception_flags':   exception_flags,
        'dates':             dates,
        'n_exceptions':      int(N),
        'n_observations':    int(T),
        'exception_rate':    round(N / T * 100, 2),
        'expected_rate':     round(p0 * 100, 2),
        'kupiec_lr':         round(LR, 4),
        'kupiec_pvalue':     round(p_value, 4),
        'kupiec_reject_h0':  p_value < 0.05,
        'basel_zone':        zone,
        'zone_color':        zone_color,
        'max_exception_loss': round(float(np.min(actual_pnl[exception_flags])), 2) if N > 0 else 0,
        'avg_exception_loss': round(float(actual_pnl[exception_flags].mean()), 2) if N > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════════
# FULL PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_full_analysis(prices_path='prices.csv',
                      weights=None,
                      portfolio_value=10_000_000,
                      n_simulations=10_000):
    """Run all 5 engines and return consolidated results dict."""
    prices, returns = load_data(prices_path)
    n_assets = len(returns.columns)

    if weights is None:
        weights = [1/n_assets] * n_assets  # equal weight default

    # print(f"Portfolio: ₹{portfolio_value:,} across {n_assets} assets")
    # print(f"Weights: {[round(w,3) for w in weights]}")
    # print(f"Data: {len(returns)} days of returns\n")

    # print("Engine 1: Monte Carlo simulation...")
    mc = monte_carlo_simulation(returns, weights, portfolio_value, n_simulations)

    # print("Engine 2: VaR + Expected Shortfall...")
    var_es = calculate_var_es(mc, returns, weights, portfolio_value,
                               confidence_levels=(0.95, 0.99))

    # print("Engine 3: Stress testing...")
    stress = run_stress_tests(weights, portfolio_value)

    # print("Engine 4: Gaussian copula...")
    copula = gaussian_copula_simulation(returns, weights, portfolio_value, n_simulations)

    # print("Engine 5: VaR backtesting (Kupiec)...")
    backtest = backtest_var(returns, weights, portfolio_value)

    # print(f"\n{'='*50}")
    # print(f"RESULTS SUMMARY")
    # print(f"{'='*50}")
    # print(f"1-Day VaR (99%, Historical):  ₹{var_es[0.99]['var_historical']:>12,.0f}")
    # print(f"1-Day VaR (99%, Monte Carlo): ₹{var_es[0.99]['var_montecarlo']:>12,.0f}")
    # print(f"1-Day ES  (99%, Monte Carlo): ₹{var_es[0.99]['es_montecarlo']:>12,.0f}")
    # print(f"Copula VaR (99%):             ₹{copula['copula_var_99']:>12,.0f}")
    # print(f"Tail amplification:           {copula['tail_amplification']:>11.1f}%")
    # print(f"Backtest exceptions:          {backtest['n_exceptions']:>12d} / {backtest['n_observations']}")
    # print(f"Basel zone:                   {backtest['basel_zone']:>12s}")
    # print(f"Kupiec p-value:               {backtest['kupiec_pvalue']:>12.4f}")
    worst = min(stress.items(), key=lambda x: x[1]['pnl'])
    # print(f"Worst stress scenario:        {worst[0]}")
    # print(f"Worst stress P&L:             ₹{worst[1]['pnl']:>12,.0f} ({worst[1]['pnl_pct']:.1f}%)")

    return {
        'prices': prices, 'returns': returns,
        'weights': weights, 'assets': list(returns.columns),
        'portfolio_value': portfolio_value,
        'mc': mc, 'var_es': var_es,
        'stress': stress, 'copula': copula,
        'backtest': backtest,
    }


if __name__ == '__main__':
    results = run_full_analysis()
