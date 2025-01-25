import pandas as pd
import numpy as np


def load_bbg_prices():
    bbg_prices = pd.read_excel("data/Bloomberg_Prices.xlsx")
    bbg_msci_prices = pd.read_excel(
        "data/Bloomberg_Prices.xlsx", sheet_name="MSCI_World"
    )

    return bbg_prices, bbg_msci_prices


def preprocess_serie(prices, companies_to_keep=None):
    """
    change axe to Date and delete NaN values
    """
    prices["Dates"] = pd.to_datetime(prices["Dates"])
    prices.set_index("Dates", inplace=True)

    if companies_to_keep is not None:
        prices = prices.loc[:, prices.columns.isin(companies_to_keep)]

    prices = prices.dropna(axis=1, how="any")

    # Delete problematic companies
    # prices = prices.drop(["IT0003492391"], axis=1) Already removed in companies to keep

    return prices


def get_returns(prices):
    returns = prices.pct_change(fill_method=None).dropna()
    return returns

def cov_with_nan(s1,s2):
    """
    Compute the covariance between two series with NaN values
    """
    mask = ~np.isnan(s1) & ~np.isnan(s2)
    return np.cov(s1[mask],s2[mask])[0,1]

def compute_covariance_matrix(returns, market_returns):
    """
    Compute the covariance matrix of the returns
    """

    #cov_with_market = [returns[col].cov(market_returns["MXWO Index"]) for col in returns.columns]

    cov_with_market = [cov_with_nan(returns[col],market_returns["MXWO Index"]) for col in returns.columns]

    # Variance du MSCI World
    msci_world_variance = market_returns.var()[0]

    # Calcul des betas pour chaque action
    betas = cov_with_market / msci_world_variance

    # Covariance des facteurs (Omega)
    omega = msci_world_variance

    # Variance spécifique pour chaque action (D)
    specific_variance = returns.var() - betas * omega

    # Construction de la matrice de covariance à un facteur
    # Matrice de covariance entre les actions basée sur le facteur (B * Omega * B.T)
    factor_covariance = np.outer(betas, betas) * omega

    # Matrice diagonale de variance spécifique (D)
    specific_covariance = np.diag(specific_variance)

    # Matrice totale de covariance (Sigma)
    cov_matrix_sigma = factor_covariance + specific_covariance

    return cov_matrix_sigma


if __name__ == "__main__":
    bbg_prices, bbg_msci_prices = load_bbg_prices()

    # Remove companies with missing carbon intensity
    carbon_intensity = pd.read_csv("data/CarbonIntensity.csv")

    # Remove companies with missing carbon intensity
    companies_to_keep = carbon_intensity["ISSUER_ISIN"].to_list()
    y = preprocess_serie(bbg_prices, companies_to_keep=companies_to_keep)

    # Remove companies with missing bbg prices
    companies_to_keep = y.columns.to_list()
    carbon_intensity = carbon_intensity[
        carbon_intensity["ISSUER_ISIN"].isin(companies_to_keep)
    ]
    carbon_intensity["Weight"] /= carbon_intensity["Weight"].sum()

    carbon_intensity.to_csv("data/filtered_CarbonIntensity.csv", index=False)

    returns = get_returns(y).iloc[:-1]
    market_returns = get_returns(preprocess_serie(bbg_msci_prices))

    cov_matrix_sigma = compute_covariance_matrix(returns, market_returns)
    np.save("data/cov_matrix_sigma.npy", cov_matrix_sigma)
    print(cov_matrix_sigma)

    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # corr_matrix = cov_matrix_sigma / np.outer(
    #     np.sqrt(np.diag(cov_matrix_sigma)), np.sqrt(np.diag(cov_matrix_sigma))
    # )

    # sns.heatmap(corr_matrix)
    # plt.show()
