"""Granger causality, cointegration, and copula analysis driven by config.yaml."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import scipy.stats as stats
import statsmodels.api as sm
import yaml
import yfinance as yf
from copulas.bivariate import Bivariate
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _output_path(config: dict[str, Any], relative: str) -> Path:
    base = Path(config.get("output_dir", "."))
    if not base.is_absolute():
        base = REPO_ROOT / base
    return base / relative


def _maybe_show(config: dict[str, Any]) -> None:
    if config.get("show_plots", False):
        plt.show()
    else:
        plt.close()


def adf_test(series: pd.Series, name: str) -> None:
    result = adfuller(series)
    print(f"ADF Test for {name}:")
    print(f"Test Statistic: {result[0]:.4f}")
    print(f"P-Value: {result[1]:.4f}")
    if result[1] > 0.05:
        print(f"{name} has a unit root (non-stationary).\n")
    else:
        print(f"{name} is stationary.\n")


def fetch_data_from_fred(config: dict[str, Any]) -> None:
    section = config["fred_unemployment_spending"]
    start_date = section["start_date"]
    end_date = section["end_date"]
    frames = [
        web.DataReader(entry["id"], "fred", start_date, end_date).rename(
            columns={entry["id"]: entry["column"]}
        )
        for entry in section["series"]
    ]
    df = pd.concat(frames, axis=1).reset_index().rename(columns={"DATE": "date"})
    csv_path = _output_path(config, section["csv_output"])
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")

    plot_cfg = section["plot"]
    fig, ax1 = plt.subplots(figsize=tuple(plot_cfg["figsize"]))
    ax1.set_title(plot_cfg["title"])
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Unemployment Rate (%)", color="red")
    ax1.plot(df["date"], df["unemployment_rate"], color="red")
    ax2 = ax1.twinx()
    ax2.set_ylabel("Consumer Spending (Billions)", color="blue")
    ax2.plot(df["date"], df["consumer_spending"], color="blue")
    plot_path = _output_path(config, plot_cfg["path"])
    plt.savefig(plot_path)
    print(f"Wrote {plot_path}")
    _maybe_show(config)

    for col in ["unemployment_rate", "consumer_spending"]:
        adf_result = adfuller(df[col])
        print(f"{col} ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3f}")

    df["unemployment_rate_diff"] = df["unemployment_rate"].diff()
    df["consumer_spending_diff"] = df["consumer_spending"].diff()
    for col in ["unemployment_rate_diff", "consumer_spending_diff"]:
        adf_result = adfuller(df[col].dropna())
        print(f"{col} ADF Statistic: {adf_result[0]:.3f}, p-value: {adf_result[1]:.3f}")

    maxlag = section["granger"]["maxlag"]
    print("\nGranger Causality Tests:")
    print("Does unemployment rate Granger-cause consumer spending?")
    grangercausalitytests(
        df[["consumer_spending_diff", "unemployment_rate_diff"]].dropna(),
        maxlag=maxlag,
    )
    print("\nDoes consumer spending Granger-cause unemployment rate?")
    grangercausalitytests(
        df[["unemployment_rate_diff", "consumer_spending_diff"]].dropna(),
        maxlag=maxlag,
    )


def run_copula_stock_interest(config: dict[str, Any]) -> None:
    section = config["copula_stock_interest"]
    np.random.seed(config.get("random_seed", 42))
    interest_rate_data = web.DataReader(
        section["fred_series"],
        "fred",
        start=section["start_date"],
        end=section["end_date"],
    )
    interest_rate_data.dropna(inplace=True)
    stock_data = yf.download(
        section["yfinance_ticker"],
        start=section["start_date"],
        end=section["end_date"],
    )
    stock_returns = stock_data["Close"].pct_change().dropna()
    data = pd.concat([stock_returns, interest_rate_data], axis=1, join="inner").dropna()
    data.columns = ["Stock Returns", "Interest Rates"]
    time_steps = len(data)
    u = stats.rankdata(data["Stock Returns"]) / (time_steps + 1)
    v = stats.rankdata(data["Interest Rates"]) / (time_steps + 1)
    uv = np.column_stack((u, v))
    best_copula = Bivariate.select_copula(uv)
    print(f"Selected Copula: {best_copula.copula_type.name}")
    best_copula.fit(uv)
    samples = best_copula.sample(section["n_samples"])
    u_future = samples[:, 0]
    v_future = samples[:, 1]
    returns_forecast = np.quantile(data["Stock Returns"], u_future)
    rates_forecast = np.quantile(data["Interest Rates"], v_future)
    plot_cfg = section["plot"]
    plt.figure(figsize=tuple(plot_cfg["figsize"]))
    plt.scatter(
        returns_forecast, rates_forecast, alpha=0.5, edgecolors="k", linewidths=0.5
    )
    plt.xlabel("Forecasted Stock Returns")
    plt.ylabel("Forecasted Interest Rates")
    plt.title(
        f"Forecasted Stock Returns vs. Interest Rates "
        f"({best_copula.copula_type.name.capitalize()} Copula)"
    )
    plt.grid(False)
    plot_path = _output_path(config, plot_cfg["path"])
    plt.savefig(plot_path)
    print(f"Wrote {plot_path}")
    _maybe_show(config)


def run_copula_inflation_unemployment(config: dict[str, Any]) -> None:
    section = config["copula_inflation_unemployment"]
    np.random.seed(config.get("random_seed", 42))
    inflation = web.DataReader(
        section["inflation_series"],
        "fred",
        start=section["start_date"],
        end=section["end_date"],
    )
    unemployment = web.DataReader(
        section["unemployment_series"],
        "fred",
        start=section["start_date"],
        end=section["end_date"],
    )
    data = pd.concat([inflation, unemployment], axis=1, join="inner").dropna()
    data.columns = ["Inflation", "Unemployment"]
    n = len(data)
    u = stats.rankdata(data["Inflation"]) / (n + 1)
    v = stats.rankdata(data["Unemployment"]) / (n + 1)
    uv = np.column_stack((u, v))
    best_copula = Bivariate.select_copula(uv)
    print(f"Best copula selected: {best_copula.copula_type.name}")
    best_copula.fit(uv)
    samples = best_copula.sample(section["n_samples"])
    u_future = samples[:, 0]
    v_future = samples[:, 1]
    inflation_forecast = np.quantile(data["Inflation"], u_future)
    unemployment_forecast = np.quantile(data["Unemployment"], v_future)
    plot_cfg = section["plot"]
    plt.figure(figsize=tuple(plot_cfg["figsize"]))
    plt.scatter(
        inflation_forecast,
        unemployment_forecast,
        alpha=0.6,
        edgecolors="k",
        linewidths=0.5,
    )
    plt.xlabel("Forecasted Inflation Rate (%)")
    plt.ylabel("Forecasted Unemployment Rate (%)")
    plt.title(
        f"Inflation vs. Unemployment Forecast "
        f"({best_copula.copula_type.name.capitalize()} Copula Model)"
    )
    plt.grid(False)
    plot_name = plot_cfg["path_template"].format(
        copula=best_copula.copula_type.name.lower()
    )
    plot_path = _output_path(config, plot_name)
    plt.savefig(plot_path)
    print(f"Wrote {plot_path}")
    _maybe_show(config)


def load_your_dataset(config: dict[str, Any]) -> None:
    section = config["wrp_dataset"]
    file_path = _output_path(config, section["file_path"])
    if not file_path.is_file():
        print(f"Skipping WRP analysis: data file not found at {file_path}")
        return

    wrp_data = pd.read_csv(file_path)
    print(wrp_data.info())
    print(wrp_data.head())

    country = section["country"]
    cols = section["columns"]
    country_data = wrp_data[wrp_data["name"] == country][
        [cols["year"], cols["christian"], cols["muslim"]]
    ].dropna()
    country_data = country_data.sort_values(cols["year"])

    plt.figure(figsize=(10, 6))
    plt.plot(
        country_data[cols["year"]],
        country_data[cols["christian"]],
        label="% Christian",
    )
    plt.plot(
        country_data[cols["year"]],
        country_data[cols["muslim"]],
        label="% Muslim",
    )
    plt.title(f"Religious Population Trends in {country}")
    plt.xlabel("Year")
    plt.ylabel("Population (%)")
    plt.legend()
    plt.grid()
    _maybe_show(config)

    adf_test(country_data[cols["christian"]], "Christian Population")
    adf_test(country_data[cols["muslim"]], "Muslim Population")

    coint_stat, p_value, critical_values = coint(
        country_data[cols["christian"]], country_data[cols["muslim"]]
    )
    print("Engle-Granger Cointegration Test:")
    print(f"Test Statistic: {coint_stat:.4f}")
    print(f"P-Value: {p_value:.4f}")
    print(f"Critical Values: {critical_values}")
    if p_value < 0.05:
        print("The two series are cointegrated.")
    else:
        print("The two series are not cointegrated.")

    X = sm.add_constant(country_data[cols["muslim"]])
    y = country_data[cols["christian"]]
    model = sm.OLS(y, X).fit()
    print(model.summary())

    plt.figure(figsize=(10, 6))
    plt.plot(country_data[cols["year"]], model.resid, label="Residuals")
    plt.axhline(0, linestyle="--", color="red", label="Zero Line")
    plt.title("Residuals of Linear Regression (% Christian ~ % Muslim)")
    plt.xlabel("Year")
    plt.ylabel("Residual")
    plt.legend()
    plt.grid()
    _maybe_show(config)
    adf_test(model.resid, "Regression Residuals")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Granger/copula analysis from YAML.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.yaml (default: beside this script)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)

    if config.get("fred_unemployment_spending", {}).get("enabled", True):
        fetch_data_from_fred(config)
    if config.get("copula_stock_interest", {}).get("enabled", True):
        run_copula_stock_interest(config)
    if config.get("copula_inflation_unemployment", {}).get("enabled", True):
        run_copula_inflation_unemployment(config)
    if config.get("wrp_dataset", {}).get("enabled", True):
        load_your_dataset(config)


if __name__ == "__main__":
    main()
