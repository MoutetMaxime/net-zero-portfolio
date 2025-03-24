import numpy as np

from PortfolioToolboxQP.PortfolioToolboxQP.Qp_Solver import qp_solver_cpp as solve


def decarbonization_pathway(t0, t, Rm, dR):
    """
    Compute the decarbonization budget with equation (1).

    param t0: base year
    param t: year index
    param Rm: minimum carbon intensity reduction
    param dR: year-to-year self decarbonization on average per annum
    """
    return 1 - (1 - dR) ** (t - t0) * (1 - Rm)


def solve_optim(R_, CI, CI0, bench, sigma, Green=None, CMstar=None, g=None, constraints_green=False, name="CE"):
    nb_firms = bench.shape[0]
    nb_years = len(R_)

    x_list = np.zeros((nb_years, nb_firms, 1))
    te = np.zeros((nb_years, 1))


    print("Sarting optimization...")
    for year in range(nb_years):
        R = R_[year]

        # Current year
        ye = 23 + year
        print(ye)

        # Extract the carbons emissions and momentum for the current year
        column = "CARBON_EMISSIONS_SCOPE_12_FY"+str(ye)
        CI_year = CI[column].values.reshape(-1, 1)
        CI_year[CI_year < 0] = 0
        column_CM = "CARBON_MOMENTUM_SCOPE_12_FY"+str(ye)
        CM_year = CI[column_CM].values.reshape(-1, 1)

        # Solve the optimization problem
        if constraints_green:
            y = solve(
                Q=sigma,
                p=None,
                G=np.concatenate((CI_year, -Green, CM_year), axis=1).T, # CI0 -Gt CM
                h=np.concatenate(((1 - R)*CI0.T@bench-CI_year.T@bench,-g*Green.T@bench, CMstar-CM_year.T@bench),axis=0),# (1-R)CI0b-CItb   -(1+g)G0b-Gtb    CM*-CMb
                A=np.ones((nb_firms,1)).T,
                b=-np.ones(bench.shape).T @ bench + 1,    
                lb=- bench,
                ub=np.ones(bench.shape) - bench, #9 * bench ?
            )
        else:
            y = solve(
                Q=sigma,
                p=None,
                G=CI_year[:,0].T,                             
                h=(1 - R) * CI0.T @ bench - CI_year.T @ bench, 
                A=np.ones((nb_firms, 1)).T,
                b=-np.ones(bench.shape).T @ bench + 1,           
                lb=- bench,
                ub=np.ones(bench.shape) - bench, #9 * bench ?
            )
        
        y = y.reshape(-1, 1)    # (_, 1)
        x = y + bench

        # Save the results
        x_list[year] = x
        name_col = "Weights_"+name+"_FY" +str(ye)
        CI[name_col] = x
        te[year] = 0.5 * y.T @ sigma @ y
    return x_list, te, CI 

def solve_optim_lambda(R_, CI, CI0, bench, sigma, Green=None, CMstar=None, g=None, l=0):
    nb_firms = bench.shape[0]
    nb_years = len(R_)

    x_list = np.zeros((nb_years, nb_firms, 1))
    te = np.zeros((nb_years, 1))

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

        y = y.reshape(-1, 1)
        x = y + bench

        x_list[year] = x
        name_col = "Weight_G_lambda_" +str(l)+ "_FY"+str(ye)
        CI[name_col] = x
        te[year] = 0.5 * y.T @ sigma @ y
    return x_list, te, CI