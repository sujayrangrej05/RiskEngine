"""
Flask backend for RiskEngine
Provides a simple API that runs the risk analysis and returns a JSON summary.
Serves a static frontend located in the ./frontend directory.
"""

from flask import Flask, jsonify, send_from_directory
from risk_engines import run_full_analysis
from charts import fig_loss_distribution, fig_var_comparison, fig_copula_comparison
import os

app = Flask(__name__, static_folder='frontend', static_url_path='')

@app.route('/')
def index():
    # Serve the index.html from the frontend folder
    return app.send_static_file('index.html')

@app.route('/api/run')
def api_run():
    """Run the full analysis and return key results as JSON."""
    results = run_full_analysis()
    var_es = results['var_es']
    copula = results['copula']
    backtest = results['backtest']
    # Generate figures JSON
    loss_fig = fig_loss_distribution(results['mc'], var_es)
    loss_json = loss_fig.to_json()
    var_comp_fig = fig_var_comparison(var_es)
    var_comp_json = var_comp_fig.to_json()
    copula_fig = fig_copula_comparison(copula, var_es)
    copula_json = copula_fig.to_json()
    summary = {
        "var_historical_99": var_es[0.99].get('var_historical'),
        "var_montecarlo_99": var_es[0.99].get('var_montecarlo'),
        "es_montecarlo_99": var_es[0.99].get('es_montecarlo'),
        "copula_var_99": copula.get('copula_var_99'),
        "tail_amplification": copula.get('tail_amplification'),
        "backtest_exceptions": backtest.get('n_exceptions'),
        "backtest_observations": backtest.get('n_observations'),
        "basel_zone": backtest.get('basel_zone'),
        "kupiec_pvalue": backtest.get('kupiec_pvalue'),
        "loss_distribution": loss_json,
        "var_comparison": var_comp_json,
        "copula_comparison": copula_json,
    }
    return jsonify(summary)

if __name__ == '__main__':
    # Run on 0.0.0.0 so it is reachable from the container/browser
    app.run(debug=True, host='0.0.0.0', port=5000)
