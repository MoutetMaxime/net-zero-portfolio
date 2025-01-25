import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def load_msci():
    datamsci = pd.read_csv("data/DataMSCI.csv", sep=";")
    datamsci = datamsci[~datamsci["ISSUER_ISIN"].isin(["IT0003492391", "AU000000REH4", "GB00BMX86B70","AU000000SVW5",'JP3538800008', 'JP3502200003', 'US30231G1022', 'NO0005052605', 'US9426222009', 'JP3902400005', 'JP3435000009', 'US4592001014', 'JP3866800000', 'GB0007980591', 'CH0012138605', 'DE0007664005', 'DE0005140008', 'JP3258000003', 'US46625H1005', 'CA4530384086', 'JP3201200007', 'GB0033195214', 'SE0000108227', 'CH0038863350', 'NL0000009538', 'US4595061015', 'US5801351017', 'JP3854600008', 'BE0003604155', 'DE0008404005', 'US02209S1033', 'US0258161092', 'LU1598757687', 'JP3942400007', 'ES0113900J37', 'US1941621039', 'DE0007100000', 'US5324571083', 'US3453708600', 'GB0005405286', 'JP3788600009', 'US6516391066', 'US7170811035', 'US7427181091', 'JP3973400009', 'GB0007188757', 'CH0012032048', 'DE0007236101', 'JP3621000003', 'SE0000108656', 'ES0178430E18', 'US90459E1064', 'US25240J1051', 'US7181721090', 'AU000000CAR3', 'US00287Y1091', 'CA0679011084', 'BE0974464977'])]
    datamsci = datamsci.drop(
        columns=["ISSUER_NAME", "ISSUERID", "ISSUER_LEI", "ISSUER_TICKER"]
    )

    datamsci = datamsci.dropna(subset=["CARBON_EMISSIONS_SCOPE_12_FY09"])
    datamsci = datamsci.dropna(subset=["CARBON_EMISSIONS_SCOPE_12_FY14"])
    datamsci = datamsci.dropna(subset=["CARBON_EMISSIONS_SCOPE_12_FY22"])
    datamsci = datamsci.dropna(subset=["SALES_USD_FY09"])
    datamsci = datamsci.dropna(subset=["SALES_USD_FY23"])

    return datamsci


def compute_beta_sales(row, datamsci):
    """
    Compute regression coefficient beta1 for sales for a given row (company) in the dataset.
    """
    sales_columns = [col for col in datamsci.columns if col.startswith("SALES_USD_FY")]
    years = np.arange(2009, 2024, 1).reshape(-1, 1)  # Années
    sales = row[sales_columns].values.reshape(-1, 1)  # Chiffres d'affaires

    if pd.isna(row["SALES_USD_FY23"]):
        years = years[:-1]
        sales = sales[:-1]

    model = LinearRegression().fit(years, sales)
    return model.coef_[0][0]


def compute_beta_emission(row, datamsci):
    """
    Compute regression coefficient beta1 for carbon emissions for a given row (company) in the dataset.
    """
    emission_columns = [
        col
        for col in datamsci.columns
        if col.startswith("CARBON_EMISSIONS_SCOPE_12_FY")
    ]
    years = np.arange(2009, 2024, 1).reshape(-1, 1)  # Années
    emissions = row[emission_columns].values.reshape(
        -1, 1
    )  # Émissions pour une entreprise
    if pd.isna(row["CARBON_EMISSIONS_SCOPE_12_FY23"]):
        years = years[:-1]
        emissions = emissions[:-1]

    model = LinearRegression().fit(years, emissions)
    return model.coef_[0][0]  # Le coefficient beta1


def project_emissions(row, datamsci):
    """
    Project carbon emissions for the years 2023 to 2050.
    """

    beta1 = row["Beta1_emissions"]

    emission_columns = [
        col
        for col in datamsci.columns
        if col.startswith("CARBON_EMISSIONS_SCOPE_12_FY")
    ]
    last_valid_year = max(
        [
            int(col.split("_FY")[-1]) + 2000
            for col in emission_columns
            if not pd.isna(row[col])
        ]
    )

    # Si 2023 est manquant, calculer sa projection
    if pd.isna(row["CARBON_EMISSIONS_SCOPE_12_FY23"]):
        row["CARBON_EMISSIONS_SCOPE_12_FY23"] = row[
            f"CARBON_EMISSIONS_SCOPE_12_FY{last_valid_year % 100:02d}"
        ] + beta1 * (2023 - last_valid_year)

    # Projeter les émissions pour les années 2024 à 2050
    for year in range(2024, 2051):
        row[f"CARBON_EMISSIONS_SCOPE_12_FY{year % 100:02d}"] = row[
            "CARBON_EMISSIONS_SCOPE_12_FY23"
        ] + beta1 * (year - 2023)

    return row


def project_sales(row):
    """
    Project sales for the years 2023 to 2050.
    """
    beta1_sales = row["Beta1_sales"]

    # Projeter les chiffres d'affaires pour les années 2024 à 2050
    for year in range(2024, 2051):
        row[f"SALES_USD_FY{year % 100:02d}"] = row["SALES_USD_FY23"] + beta1_sales * (
            year - 2023
        )

    return row

def compute_carbon_momentum(df):
    """
    Compute Carbon Momentum for each year starting from 2023.
    Carbon Momentum = Carbon Emissions for the year / Beta1_emissions
    We assume beta1_hat(t) dosen't change over time.
    """
    emission_columns = [
        col for col in df.columns if col.startswith("CARBON_EMISSIONS_SCOPE_12_FY")
    ]

    for year in range(2023, 2051):
        emission_col = f"CARBON_EMISSIONS_SCOPE_12_FY{year % 100:02d}"
        if emission_col in df.columns:
            df[f"CARBON_MOMENTUM_SCOPE_12_FY{year % 100:02d}"] = (
                df[emission_col] / df["Beta1_emissions"]
            )
    return df

def plot_emissions(row, datamsci):
    # Identifiez les colonnes d'émissions
    emission_columns = [
        col
        for col in datamsci.columns
        if col.startswith("CARBON_EMISSIONS_SCOPE_12_FY")
    ]
    # Années de 2009 à 2050
    years = np.arange(2009, 2051)
    # Émissions correspondant à ces années
    emissions = row[emission_columns].values

    # Années de la régression : 2009 à 2023
    regression_years = np.arange(2009, 2024).reshape(-1, 1)
    regression_emissions = row[emission_columns[: len(regression_years)]].values

    # Ajuster le modèle de régression linéaire
    model = LinearRegression().fit(regression_years, regression_emissions)
    regression_line = model.predict(regression_years)

    # Séparer les années pour les projections (2024 à 2050)
    projection_years = years[15:]  # Années 2024 à 2050
    projection_emissions = emissions[15:]

    # Tracer les données réelles (2009 à 2023)
    plt.plot(
        years[:15],
        emissions[:15],
        label="Données réelles (2009-2023)",
        color="blue",
        marker="o",
    )

    # Tracer la droite de régression pour 2009 à 2023
    plt.plot(
        regression_years,
        regression_line,
        linestyle="--",
        color="blue",
        label="Régression (2009-2023)",
    )

    # Tracer les projections (2024 à 2050)
    plt.plot(
        projection_years,
        projection_emissions,
        label="Projections (2024-2050)",
        color="red",
        marker="o",
    )

    # Ajouter des labels et légendes
    plt.xlabel("Année")
    plt.ylabel("Émissions de carbone")
    plt.title(f"Projection des émissions de carbone ({row['ISSUER_ISIN']})")
    plt.legend()
    plt.grid(True)
    plt.show()


def compute_emissions_sales(df):
    df["Beta1_emissions"] = df.apply(lambda x: compute_beta_emission(x, df), axis=1)
    df["Beta1_sales"] = df.apply(lambda x: compute_beta_sales(x, df), axis=1)

    # Project Carbon emission
    df = df.apply(lambda x: project_emissions(x, df), axis=1)

    # Project Sales
    df = df.apply(project_sales, axis=1)

    return df


def compute_intensity(df):

    # ce3_columns = [
    #     col for col in df.columns if col.startswith("CARBON_EMISSIONS_SCOPE_3")
    # ]
    ce12_columns = [
        col for col in df.columns if col.startswith("CARBON_EMISSIONS_SCOPE_12")
    ]
    sales_columns = [col for col in df.columns if col.startswith("SALES")]

    for i, year in enumerate(range(2009, 2051)):
        df[f"CI_Scope12_FY{year % 100:02d}"] = (
            df[ce12_columns[i]] / df[sales_columns[i]]
        )
        # df[f"CI_Scope3_FY{year % 100:02d}"] = df[ce3_columns[i]] / df[sales_columns[i]]

    return df


if __name__ == "__main__":
    df = load_msci()
    weights = df[["ISSUER_ISIN", "MarketCap_USD"]]
    weights["Weight"] = weights["MarketCap_USD"] / weights["MarketCap_USD"].sum()
    df = compute_emissions_sales(df)
    df = compute_intensity(df)
    df = pd.merge(df, weights, on="ISSUER_ISIN", how="left")
    df = df[
        ["ISSUER_ISIN","GICS_SUB_IND", "GICS_SECTOR","EST_EU_TAXONOMY_MAX_REV",
         "EU_TAXONOMY_ADAPTATION_ELIGIBLE_MAX_REV","EU_TAXONOMY_MITIGATION_ELIGIBLE_MAX_REV","GICS_SECTOR"]
         # We capture the greenness with "EST_EU_TAXONOMY_MAX_REV"
         # We can put it in the context of the industry with "EU_TAXONOMY_ADAPTATION_ELIGIBLE_MAX_REV","EU_TAXONOMY_MITIGATION_ELIGIBLE_MAX_REV"
         # a / ((b+c)/2)
        + ["Weight"]
        + [col for col in df.columns if col.startswith("CI_Scope12_FY")]
    ]

    print(df["Weight"].sum())

    df.to_csv("data/CarbonIntensity.csv", index=False)
