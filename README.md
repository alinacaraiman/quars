# quars
Robust tools for finance, econometrics, econophysics, and quantitative analysis in rust.

Currently his repository hosts only an early implementation of a risk-adjusted mean‐variance portfolio optimizer in Rust.

## Overview

**Key Features (MVP)**

1. **Data Collection**: Fetches or reads historical price data (CSV or placeholder for API).
2. **Portfolio Statistics**: Computes historical returns and a  covariance matrix.
3. **Optimization (Risk-Adjusted Return Maximization)**:
  The current optimization method (Mean Variance Optimization) finds the portfolio that maximizes the risk-adjusted return. Aditionally, for demonstrative and educational purposes, you can use the near-optimality method (Marin Lolic, 2024) denoted in this [paper](https://www.mdpi.com/1911-8074/17/5/183). The risk-adjusted optimization is formalized as:

  $$
  \begin{aligned}
  &\max_{\mathbf{x}} \quad (\boldsymbol{\mu} - r_f\,\mathbf{1})^T \mathbf{x} - \tau \, \mathbf{x}^T \boldsymbol{\Sigma} \,\mathbf{x} \\
  &\text{subject to} \quad \mathbf{1}^T \mathbf{x} = 1,
  \end{aligned}
  $$

  where:
  - $$\( \boldsymbol{\mu} \)$$ is the vector of expected returns of the risky assets,
  - $$\( r_f \)$$ is the risk-free rate,
  - $$\( \boldsymbol{\Sigma} \)$$ is the covariance matrix of returns,
  - $$\( \mathbf{x} \)$$ is the vector of portfolio weights, and
  - $$\( \tau \)$$ is the risk-aversion parameter.

4. **Efficient Frontier & Visualization:**
  Generates plots for the efficient frontier and the capital allocation line (CAL) that incorporate the risk-free asset, along with a separate visualization of the portfolio return distribution with VaR and CVaR thresholds.

---
## Getting Started

1. **Clone the Repo & Enter Directory**

   ```bash
   git clone https://github.com/alinacaraiman/quars.git
   cd quars
   ```
2. **Set the API Key of your preffered data broker**: Before running quars, you must set your API key as an environment variable. Currently only Alpha Vantage and Twelve API supported. Please refer from using any other variable name than **APP__DATA_API__API_KEY**. Example `.env` file:
   ```dotenv
   APP__DATA_API__API_KEY=your_data_api_key_here
   ```
3. **Configuration**: Quars requires specific settings in a configuration file `config.toml` to control how data is accessed and processed:
   ```toml
   [general]
   data_source = "api"            # Use the API to fetch data, or "csv" to read from file.
   data_file = "data/historical_data.csv"  # Path to the CSV file, if using CSV.

   [portofolio_optimization]
   method = "MVO"                 # The general method to be used for the optimization, currently only Mean-Variance Optimization supported
   sub_method = "near-optimal" # Currently "risk-adjusted" and "near-optimal" supported
   risk_free_rate = 0.025         # The risk-free rate, used for portfolio optimization.
   params = [0.1]                 # Depending on the chosen submethod, you can define a vector of parameters (e.g. Near-Optimality Method uses 2 parameters, Tau (risk-aversion parameter) and Theta (concentration parameter))

   [data_api]
   source = "twelve"              # Specify the data broker ("  twelve" for Twelve Data, "alphavantage", etc.)
   tickers = ["AAPL", "GOOGL"]      # List of ticker symbols to fetch data for.
   start_date = "2020-01-01"        # Start date for historical data (YYYY-MM-DD format).
   end_date = "2020-12-31"          # End date for historical data (YYYY-MM-DD format).
   timeframe = "daily"            # Time interval for data ("5min", "daily", "weekly", "monthly", etc.)

   ```
## OpenBLAS
To run properly some parts of this repo, you need to install OpenBLAS. For more [information](https://github.com/blas-lapack-rs/openblas-src) on OpenBLAS in Rust.

### Windows
**Install OpenBLAS via vcpkg:**
   - To install the dynamic version (which provides `libopenblas.dll` along with LAPACK routines) using vcpkg:
     ```powershell
     vcpkg install openblas:x64-windows
     ```
   - Alternatively, for a static build (eliminating the DLL dependency):
     ```powershell
     vcpkg install openblas:x64-windows-static
     ```
   - If you use the dynamic version, note the DLL is installed in:
     ```
     path\to\vcpkg\installed\x64-windows\bin
     ```
**Set up Environment Variables:**
   - **VCPKG_ROOT:**
     Ensure that if you set this variable, it uses the correct path. For example, in PowerShell:
     ```powershell
     $env:VCPKG_ROOT = "path\to\vcpkg"
     ```
   - **PATH (for dynamic linking):**
     Add the vcpkg bin folder to your PATH so Windows can find `libopenblas.dll`:
     ```cmd
     set PATH=path\to\vcpkg\installed\x64-windows\bin;%PATH%
     ```

### For Linux/macOS

On Linux or macOS, you can use your package manager to install OpenBLAS, which already includes LAPACK routines. Alternatively refer [here](https://github.com/blas-lapack-rs/openblas-src)

1. **Install OpenBLAS:**
   - **Ubuntu/Debian:**
     ```bash
     sudo apt-get update
     sudo apt-get install libopenblas-dev
     ```
   - **Fedora:**
     ```bash
     sudo dnf install openblas-devel
     ```
   - **macOS (with Homebrew):**
     ```bash
     brew install openblas
     ```
   - These packages typically include the LAPACK routines required.