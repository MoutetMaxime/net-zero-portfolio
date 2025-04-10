import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math 
from PortfolioToolboxQP.PortfolioToolboxQP.Qp_Solver import qp_solver_cpp as solve
from mpl_toolkits.mplot3d import Axes3D

def decarbonization_pathway(t0, t, Rm, dR):
    """
    Compute the decarbonization budget with equation (1).

    param t0: base year
    param t: year index
    param Rm: minimum carbon intensity reduction
    param dR: year-to-year self decarbonization on average per annum
    """
    return 1 - (1 - dR) ** (t - t0) * (1 - Rm)

def te_over_time(te_array, time,title=''):
    plt.figure(figsize=(10, 6))
    plt.plot(time, te_array, marker='o', linestyle='-', color='b')
    plt.title('Tracking Error Over Time'+title)
    plt.xlabel('Year')
    plt.ylabel('Tracking Error')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def hist_per_sector(CI, name_col='Weight_CE_FY',title=''):
    bench = (CI["Weight"].values / CI["Weight"].sum()).reshape(-1, 1)
    CI["benchmark"] = bench
    grouped = CI.groupby("GICS_SECTOR")[[ "benchmark",name_col+"23",name_col+"50"]].sum()
    # Tracer l'histogramme
    grouped.plot(
        kind="bar",
        figsize=(12, 5),
        stacked=False,  # Changez en True pour un histogramme empilé
    )
    plt.title("Somme des colonnes par secteur"+title)
    plt.xlabel("Secteur")
    plt.ylabel("Somme")
    plt.xticks(rotation=45, ha="right")  # Rotation des étiquettes pour lisibilité
    plt.legend(["Benchmark","Weight_FY23", "Weight_FY50"], loc="upper right")  # Légende personnalisée
    plt.tight_layout()
    plt.show()

def evolution_weights_per_sector(CI,name_col='Weight_CE_FY',title=''):
    # Extract the columns related to weights over the years
    weight_columns = [col for col in CI.columns if col.startswith(name_col)]

    # Group by sector and sum the weights for each year
    sector_weights = CI.groupby('GICS_SECTOR')[weight_columns].sum()

    # Plot the evolution of weights per sector over the years
    sector_weights.T.plot(figsize=(12, 4), marker='o')
    plt.title('Evolution of Weights per Sector Over the Years'+title)
    plt.xlabel('Year')
    plt.ylabel('Weight')
    plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(ticks=range(len(weight_columns)), labels=[col.split('_')[-1] for col in weight_columns], rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evolution_of_weights_vs_bench(CI,name_col='Weight_CE_FY',title=''):
    # Extract the columns related to weights over the years
    weight_columns = [col for col in CI.columns if col.startswith(name_col)] 

    # Group by sector and sum the weights for each year
    sector_weights = CI.groupby('GICS_SECTOR')[weight_columns].sum()
    benchmark_val = CI.groupby('GICS_SECTOR')["benchmark"].sum()
        # Number of sectors
    num_sectors = len(sector_weights.index)

    # Define the number of columns for subplots (e.g., 3 columns per row)
    cols = 3
    rows = math.ceil(num_sectors / cols)

    # Create subplots
    fig, axes = plt.subplots(rows, cols, figsize=(15, 10), constrained_layout=True)
    axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing

    # Plot for each sector
    for i, sector in enumerate(sector_weights.index):
        ax = axes[i]
        
        # Plot the weights over the years for the current sector
        ax.plot(
            [col.split('_')[-1] for col in weight_columns],  # Extract years from column names
            sector_weights.loc[sector],
            marker='o',
            label='Weights'
        )
        
        # Add a horizontal line for the benchmark value of the current sector
        ax.axhline(
            y=benchmark_val.loc[sector], 
            color='r', 
            linestyle='--', 
            label='Benchmark'
        )
        
        # Titles and labels
        ax.set_title(sector+title, fontsize=10)
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel('Weight', fontsize=8)
        ax.tick_params(axis='both', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.show()

def evolution_of_non_zeros_per_sector_normalized(CI, name_col='Weight_CE_FY',title=''):
    # Extraire les colonnes liées aux poids au fil des années
    weight_columns = [col for col in CI.columns if col.startswith(name_col)]

    # Ajouter des colonnes "zero_" indiquant si les valeurs sont significativement différentes de zéro
    for i, col in enumerate(weight_columns):
        CI[f"zero_{23 + i}"] = CI[col] > 0.00001  # Ajuster l'index des années (par exemple, FY23 = 2023)

    # Sélectionner les colonnes "zero_" pour le comptage
    zero_columns = [col for col in CI.columns if col.startswith('zero_')]

    # Grouper par secteur et compter les valeurs "True" pour chaque année
    sector_counts = CI.groupby('GICS_SECTOR')[zero_columns].sum()

    # Calculer le nombre total de lignes par secteur
    sector_totals = CI.groupby('GICS_SECTOR').size()

    # Diviser chaque somme par le nombre de lignes du secteur pour obtenir les parts
    sector_proportions = sector_counts.div(sector_totals, axis=0)
    # Transposer pour visualiser les années sur l'axe X
    sector_proportions.T.plot(figsize=(14, 5), marker='o')

    # Ajouter des titres, labels et personnalisation
    plt.title('Evolution of Non-Zero Weight Proportions per Sector Over the Years'+title, fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Proportion of Non-Zero Values')
    plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xticks(
        ticks=range(len(zero_columns)), 
        labels=[col.split('_')[-1] for col in zero_columns], 
        rotation=45, 
        fontsize=12
    )
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def evolution_of_non_zeros_per_sector(CI, name_col='Weight_CE_FY',title=''):
    # Extraire les colonnes liées aux poids au fil des années
    weight_columns = [col for col in CI.columns if col.startswith(name_col)]

    # Ajouter des colonnes "zero_" indiquant si les valeurs sont significativement différentes de zéro
    for i, col in enumerate(weight_columns):
        CI[f"zero_{23 + i}"] = CI[col] > 0.00001
        1  # Ajuster l'index des années (par exemple, FY23 = 2023)

    # Sélectionner les colonnes "zero_" pour le comptage
    zero_columns = [col for col in CI.columns if col.startswith('zero_')]

    # Grouper par secteur et compter les valeurs "True" pour chaque année
    non_zero_counts = CI.groupby('GICS_SECTOR')[zero_columns].sum()
    # Transposer pour visualiser les années sur l'axe X
    non_zero_counts.T.plot(figsize=(14, 5), marker='o')

    # Ajouter des titres, labels et personnalisation
    plt.title('Evolution of Non-Zero Weights per Sector Over the Years'+title, fontsize=16)
    plt.xlabel('Year', fontsize=14)
    plt.ylabel('Count of Non-Zero Values')
    plt.legend(title='Sector', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    plt.xticks(
        ticks=range(len(zero_columns)), 
        labels=[col.split('_')[-1] for col in zero_columns], 
        rotation=45, 
        fontsize=12
    )
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

def greenness(Green,CI,times,name_col="Weights_CE_FY",title=''):
    weights = [col for col in CI.columns if col.startswith(name_col)]
    x_list = [CI[weight].values for weight in weights]
    green_plot = [Green.T @ x for x in x_list] 
    green_plot = np.array(green_plot).squeeze()
    plt.figure(figsize=(10, 6))
    plt.plot(times,green_plot,marker='o', linestyle='-', color='g')
    plt.title('Greenness Over Time' + title)
    plt.xlabel('Year')
    plt.ylabel('Greenness')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def spider_graph(CI, name_col='Weight_CE_FY',title=''):
    categories =  CI["GICS_SECTOR"].unique()
    # Extraire les colonnes liées aux poids au fil des années
    weight_columns = [col for col in CI.columns if col.startswith(name_col)]

    # Ajouter des colonnes "zero_" indiquant si les valeurs sont significativement différentes de zéro
    for i, col in enumerate(weight_columns):
        CI[f"zero_{23 + i}"] = CI[col] > 0.00001  # Ajuster l'index des années (par exemple, FY23 = 2023)

    # Sélectionner les colonnes "zero_" pour le comptage
    zero_columns = [col for col in CI.columns if col.startswith('zero_')]

    # Grouper par secteur et compter les valeurs "True" pour chaque année
    sector_counts = CI.groupby('GICS_SECTOR')[zero_columns].sum()

    # Calculer le nombre total de lignes par secteur
    sector_totals = CI.groupby('GICS_SECTOR').size()

    # Diviser chaque somme par le nombre de lignes du secteur pour obtenir les parts
    sector_proportions = sector_counts.div(sector_totals, axis=0)
    #valeurs des proportions pour l'année 2024 ie la valeur de la colonne zero_24
    values24 = []
    values30 = []
    values40 = []
    ones = []
    for cat in categories:
        values24.append(sector_proportions.loc[cat]["zero_24"])
        values30.append(sector_proportions.loc[cat]["zero_30"])
        values40.append(sector_proportions.loc[cat]["zero_40"])
        ones.append(1)

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values24 += values24[:1]  # Idem pour les valeurs de l'année 2024
    values30 += values30[:1]  # Idem pour les valeurs de l'année 2030
    values40 += values40[:1]  # Idem pour les valeurs de l'année 2040
    ones += ones[:1]
    angles += angles[:1]  # Idem pour l'angle


    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values24, color='red', linewidth=2,label="2024")  # Contour
    ax.fill_between(angles, values30, ones, color='green', alpha=0.4)  # Surface remplie
    ax.plot(angles, values30, color='green', linewidth=2,label="2030",linestyle='dashed')  # Contour
    ax.plot(angles, values40, color='pink', linewidth=2,label="2040")  # Contour

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.title(title)
    plt.show()

def solve_optim(R_, CI, CI0, bench, sigma, Green=None, CMstar=None, g=None, constraints_green = False, name = "CE"):
    x_list = []
    te = []
    for year in range(len(R_)):
        R = R_[year]
        ye = 23+year
        print(ye)
        column = "CARBON_EMISSIONS_SCOPE_12_FY"+str(ye)
        CI_year = CI[column].values.reshape(-1, 1)
        CI_year[CI_year < 0] = 0
        column_CM = "CARBON_MOMENTUM_SCOPE_12_FY"+str(ye)
        CM_year = CI[column_CM].values.reshape(-1, 1)
        if constraints_green:
            y = solve(
                Q = sigma,
                p = None,
                G = np.concatenate((CI_year, -Green, CM_year), axis=1).T,#CI0 -Gt CM
                h = np.concatenate(((1 - R)*CI0.T@bench-CI_year.T@bench,-g*Green.T@bench, CMstar-CM_year.T@bench),axis=0),# (1-R)CI0b-CItb   -(1+g)G0b-Gtb    CM*-CMb
                A = np.ones((bench.shape[0],1)).T,
                b = -np.ones(bench.shape).T @ bench + 1,
                lb = - bench,
                ub =np.ones(bench.shape) - bench, #9 * bench ?
            )
        else:
            y = solve(
            Q = sigma,
            p = None,
            G = CI_year[:,0].T,
            h = (1 - R) * CI0.T @ bench - CI_year.T @ bench,
            A = np.ones((bench.shape[0],1)).T,
            b =-np.ones(bench.shape).T @ bench + 1,
            lb = - bench,
            ub =np.ones(bench.shape) - bench, #9 * bench ?
        )
        y = y[:,np.newaxis]
        x = y + bench
        x_list.append(x)
        name_col = "Weights_"+name+"_FY" +str(ye)
        CI[name_col] = x
        tracking_error = 0.5 * y.T @ sigma @ y
        te.append(tracking_error)
    return x_list, te, CI 

def solve_optim_lambda(R_, CI, CI0, bench, sigma, Green=None, CMstar=None, g=None,l = 0):
    x_list=[]
    te = []
    for year in range(len(R_)):
        R = R_[year]
        ye = 23+year
        column_CE = "CARBON_EMISSIONS_SCOPE_12_FY"+str(ye)
        CI_year = CI[column_CE].values.reshape(-1, 1)
        column_CM = "CARBON_MOMENTUM_SCOPE_12_FY"+str(ye)
        CM_year = CI[column_CM].values.reshape(-1, 1)
        CI_year[CI_year < 0] = 0
        y = solve(
            Q = sigma,
            p = -l*Green,
            G = np.concatenate((CI_year, -Green, CM_year), axis=1).T,
            h = np.concatenate(((1 - R)*CI0.T@bench-CI_year.T@bench,-g*Green.T@bench, CMstar-CM_year.T@bench),axis=0),
            A = np.ones((bench.shape[0],1)).T,
            b = -np.ones(bench.shape).T @ bench + 1,
            lb = - bench,
            ub =np.ones(bench.shape) - bench, #9 * bench ?
        )
        y = y[:,np.newaxis]
        x = y + bench
        x_list.append(x)
        name_col = "Weight_G_lambda_" +str(l)+ "_FY"+str(ye)
        CI[name_col] = x
        tracking_error = 0.5 * y.T @ sigma @ y
        te.append(tracking_error)
    return x_list, te, CI

def graph_3D_surface(Green, x_list_year, lambdas,years,te_plot,title=''):
    green_plot = [[(Green.T @ x)[0][0] for x in x_list_year[i]] for i in range(len(x_list_year))]
    green_plot = np.array(green_plot).squeeze()

    # Create a meshgrid for the years and lambdas
    lambdas_grid, years_grid = np.meshgrid(lambdas, years)

    # Plotting the 3D plot
    fig = plt.figure(figsize=(12, 20))
    ax1 = fig.add_subplot(121, projection='3d')

    # Plot the surface
    ax1.plot_surface(lambdas_grid, years_grid, te_plot.T, cmap='viridis')

    # Add labels and title
    ax1.set_xlabel('Lambda')
    ax1.set_ylabel('Year')
    ax1.set_zlabel('Tracking Error (sqrt(te) * 1e4)')
    ax1.set_title('3D Plot of Tracking Error vs Year and Lambda')

    ax2 = fig.add_subplot(122, projection='3d')

    # Plot the surface
    ax2.plot_surface(lambdas_grid, years_grid, green_plot.T, cmap='viridis')

    # Add labels and title
    ax2.set_xlabel('Lambda')
    ax2.set_ylabel('Year')
    ax2.set_zlabel('Tracking Error (sqrt(te) * 1e4)')
    ax2.set_title('3D Plot of Tracking Error vs Year and Lambda')
    plt.suptitle(title)
    plt.show()

def graphe_3D_block(Green, x_list_year, lambdas, years, te_plot,title=''):
    green_plot = [[(Green.T @ x)[0][0] for x in x_list_year[i]] for i in range(len(x_list_year))]
    green_plot = np.array(green_plot).squeeze()
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Positions des barres en 3D
    xpos, ypos = np.meshgrid(lambdas, years)  # Création des grilles pour lambda et année
    xpos = xpos.flatten()  # Aplatir en liste de coordonnées
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)  # Base des barres (z=0)

    # Largeur et profondeur des barres
    dx = np.abs(np.diff(lambdas).mean()) * 0.8  # Ajuste la largeur des barres
    dy = np.abs(np.diff(years).mean()) * 0.8  

    # Hauteur des barres (valeurs de te_year_array)
    dz1 = te_plot.T.flatten()
    dz2 = green_plot.T.flatten()

    # Coloration basée sur la hauteur des barres
    colors1 = plt.cm.viridis((dz1 - dz1.min()) / (dz1.max() - dz1.min()))
    colors2 = plt.cm.viridis((dz2 - dz2.min()) / (dz2.max() - dz2.min()))

    # Création de l'histogramme 3D
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz1, color=colors1, shade=True)
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz2, color=colors2, shade=True)

    # Labels et titre
    ax1.set_xlabel("Lambda")
    ax1.set_ylabel("Année")
    ax1.set_zlabel("Tracking Error")
    ax1.set_title("Histogramme 3D du Tracking Error")
    ax2.set_xlabel("Lambda")
    ax2.set_ylabel("Année")
    ax2.set_zlabel("Greenness")
    ax2.set_title("Histogramme 3D de Greenness")
    plt.suptitle(title)

    plt.show()

def graphe_3D_block_surface_ref(Green, x_list_year, lambdas, years, te_plot, te_green, x_list_green,te_year,title=''):
    lambdas_grid, years_grid = np.meshgrid(lambdas, years)
    green_plot = [[(Green.T @ x)[0][0] for x in x_list_year[i]] for i in range(len(x_list_year))]
    green_plot = np.array(green_plot).squeeze()

    # Positions des barres en 3D
    xpos, ypos = np.meshgrid(lambdas, years)  # Création des grilles pour lambda et année
    xpos = xpos.flatten()  # Aplatir en liste de coordonnées
    ypos = ypos.flatten()
    zpos = np.zeros_like(xpos)  # Base des barres (z=0)

    # Largeur et profondeur des barres
    dx = np.abs(np.diff(lambdas).mean()) * 0.8  # Ajuste la largeur des barres
    dy = np.abs(np.diff(years).mean()) * 0.8  

    # Hauteur des barres (valeurs de te_year_array)
    dz1 = te_plot.T.flatten()
    dz2 = green_plot.T.flatten()

    # Coloration basée sur la hauteur des barres
    colors1 = plt.cm.viridis((dz1 - dz1.min()) / (dz1.max() - dz1.min()))
    colors2 = plt.cm.viridis((dz2 - dz2.min()) / (dz2.max() - dz2.min()))
    te_plot_green = [[np.sqrt(te_green[j][0][0])*1e4 for j in range(len(te_year[0]))] for i in range(len(te_year))]
    te_plot_green = np.array(te_plot_green).squeeze()

    x_list_green = np.array(x_list_green).squeeze()
    green_plot_ = [[(Green.T)@(x_list_green[i]) for i in range(len(te_year[0]))] for j in range(len(x_list_year))]
    green_plot_ = np.array(green_plot_).squeeze()

    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    # Création de l'histogramme 3D
    ax1.bar3d(xpos, ypos, zpos, dx, dy, dz1, color=colors1, shade=True, alpha = 0.2)
    ax1.plot_surface(lambdas_grid, years_grid, te_plot_green.T)
    ax2.bar3d(xpos, ypos, zpos, dx, dy, dz2, color=colors2, shade=True, alpha = 0.2)
    ax2.plot_surface(lambdas_grid, years_grid, green_plot_.T)

    # Labels et titre
    ax1.set_xlabel("Lambda")
    ax1.set_ylabel("Année")
    ax1.set_zlabel("Tracking Error")
    ax1.set_title("Histogramme 3D du Tracking Error")
    ax2.set_xlabel("Lambda")
    ax2.set_ylabel("Année")
    ax2.set_zlabel("Greenness")
    ax2.set_title("Histogramme 3D de Greenness")
    plt.suptitle(title)
    plt.show()