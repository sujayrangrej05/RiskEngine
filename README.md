```
██████╗ ██╗███████╗██╗  ██╗    ███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗
██╔══██╗██║██╔════╝██║ ██╔╝    ██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝
██████╔╝██║███████╗█████╔╝     █████╗  ██╔██╗ ██║██║  ███╗██║██╔██╗ ██║█████╗  
██╔══██╗██║╚════██║██╔═██╗     ██╔══╝  ██║╚██╗██║██║   ██║██║██║╚██╗██║██╔══╝  
██║  ██║██║███████║██║  ██╗    ███████╗██║ ╚████║╚██████╔╝██║██║ ╚████║███████╗
╚═╝  ╚═╝╚═╝╚══════╝╚═╝  ╚═╝    ╚══════╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝╚══════╝
                                                                               
```

# Risk Engine

![License](https://img.shields.io/badge/License-MIT-blue.svg)
![Python](https://img.shields.io/badge/Python-3.12%2B-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3.3-black?logo=flask)
![Plotly](https://img.shields.io/badge/Plotly-2.30.0-orange?logo=plotly)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3.2-purple?logo=bootstrap)

---

## 🎯 Overview

**Risk Engine** is a lightweight, web‑based analytics platform that implements the five core quantitative market‑risk engines required by **Basel III**:

1. **Monte‑Carlo simulation** (Geometric Brownian Motion + Cholesky decomposition)
2. **VaR & Expected Shortfall** (Parametric, Historical, Monte‑Carlo)
3. **Historical stress testing** (real‑world crisis scenarios)
4. **Gaussian copula tail‑dependence model**
5. **VaR back‑testing** (Kupiec Proportion‑of‑Failures test with Basel traffic‑light zones)

The UI is built with **Bootstrap 5** (dark theme) and **Plotly.js** for interactive, high‑resolution charts. The backend is a minimal **Flask** server that runs the analysis on‑demand and returns a concise JSON payload.

---

## ✨ Features

- **One‑click analysis** – press *Run Analysis* and receive a full risk‑report instantly.
- **Four interactive charts**:
  - Loss‑distribution histogram with VaR/ES markers.
  - VaR & ES comparison across the three methodologies.
  - Gaussian‑copula VaR vs. standard VaR.
  - (Future‑proof – easy to add more visualisations).
- **Responsive KPI cards** that display the most important risk metrics.
- **Fully container‑friendly** – the Flask app can be run locally or inside Docker.
- **Extensible architecture** – risk‑engine logic lives in `risk_engines.py`; adding new engines is straightforward.

---

## 🛠️ Tech Stack

| Layer | Technology |
|------|-------------|
| **Backend** | Python 3.12, Flask 2.3, NumPy, Pandas, SciPy |
| **Risk‑engine** | Custom implementation (`risk_engines.py`), statistical functions, Monte‑Carlo, Gaussian copula |
| **Frontend** | HTML5, Bootstrap 5 (dark theme), Plotly.js (interactive charts) |
| **Data** | CSV (`prices.csv`) – daily price history for 10 Indian‑market equities |

---

## 🚀 Getting Started

### Prerequisites

- Python ≥ 3.12 (the project uses type‑hints and f‑strings).
- `pip` (or `uv`/`conda` if you prefer).

### Installation

```bash
# Clone the repository (or copy the source folder)
git clone https://github.com/your‑username/riskengine-dashboard.git
cd riskengine-dashboard

# Install required Python packages
pip install -r requirements.txt
```

### Running the application

```bash
# Start the Flask server (it will listen on http://0.0.0.0:5000)
python app.py
```

Open your browser and navigate to **http://127.0.0.1:5000**. Click the **Run Analysis** button – the UI will show the KPI cards and three polished charts.

---

## 📡 API

The only public endpoint is:

- `GET /api/run`

It triggers the complete risk‑engine pipeline (`run_full_analysis`) and returns a JSON object with:

```json
{
  "var_historical_99": <float>,
  "var_montecarlo_99": <float>,
  "es_montecarlo_99": <float>,
  "copula_var_99": <float>,
  "tail_amplification": <float>,
  "backtest_exceptions": <int>,
  "backtest_observations": <int>,
  "basel_zone": "Green|Amber|Red",
  "kupiec_pvalue": <float>,
  "loss_distribution": "{Plotly JSON}",
  "var_comparison": "{Plotly JSON}",
  "copula_comparison": "{Plotly JSON}"
}
```

The front‑end consumes this payload, builds the KPI cards, and renders the three Plotly figures.

---

## 📂 Project Structure

```
project-root/
│   app.py               # Flask server, API endpoint
│   requirements.txt      # Python dependencies
│   test_api.py          # Minimal integration test
│
├───frontend/            # Static assets served by Flask
│       index.html        # Main UI (Bootstrap + Plotly)
│       custom.css        # Theme & UI tweaks
│
├───__pycache__/          # Compiled Python files (auto‑generated)
│
└───risk_engines.py       # Core risk‑engine implementations
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Improve visualisations or add new charts.
- Extend the risk‑engine (e.g., add a GARCH model or a new stress scenario).
- Polish the UI/UX further – the current layout is a solid foundation.
- Submit bug reports or feature requests via the *Issues* tab.

Please fork the repository, create a feature branch, and open a pull request.

---

## 📄 License

This project is licensed under the **MIT License** – see the `LICENSE` file for details.

---

## 🎉 Acknowledgements

- The original Dash implementation served as a reference for the risk‑engine logic.
- Plotly.js and Bootstrap made the UI development fast and responsive.
- All data used in the demo (`prices.csv`) is public market data for illustrative purposes.

---

*Happy risk modelling!*