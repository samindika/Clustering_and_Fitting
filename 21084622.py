#Import Libraries
#import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


import sklearn.cluster as cluster
import cluster_tools as ct

from scipy.optimize import curve_fit
import scipy.optimize as opt
import errors as err
import sklearn.metrics as skmet

def read_file(file_1, file_2, file_3):
    
    #impoert Renewable_energy dataset
    df_energy = pd.read_csv(file_1,skiprows=4)
    df_energy = df_energy.drop(['Country Code', 'Indicator Code'], axis=1)
    print(df_energy.describe())

    #import GDP percapita dataset
    df_gdp = pd.read_csv(file_2,skiprows=4)
    df_gdp = df_gdp.drop(['Country Code', 'Indicator Code'], axis=1)
    print(df_gdp.describe())

    #Import CO2 emission per capita dataset
    df_co2 = pd.read_csv(file_3,skiprows=4)
    df_co2 = df_co2.drop(['Country Code', 'Indicator Code'], axis=1)
    print(df_co2.describe())
    
    return df_energy,df_gdp,df_co2


def clustering_Energy_CO2_2019(df_energy,df_gdp,df_co2):
    #Drop rows with NaNs in 2019
    
    df_energy = df_energy[df_energy["2019"].notna()]
    df_co2 = df_co2[df_co2["2019"].notna()]
    
    df_energy.describe()
    df_gdp.describe()
    df_co2.describe()
    
    #Choose only essential columns and take a copy
    df_energy_2019 = df_energy[["Country Name", "2019"]].copy()
    df_co2_2019 = df_co2[["Country Name", "2019"]].copy() 
    
    #Merge renewable energy and Co2 emission columns on Country name
    df_2019_energy_co2 = pd.merge(df_energy_2019, df_co2_2019, on="Country Name", how="outer")
    
    #Drop NaNs' in the merged dataframe
    df_2019_energy_co2 = df_2019_energy_co2.dropna() 
    
    #Rename the axis
    df_2019_energy_co2 = df_2019_energy_co2.rename(columns={"2019_x":"Renewable_energy", "2019_y":"CO2"})
    
    #Plot scatter matrix
    pd.plotting.scatter_matrix(df_2019_energy_co2, figsize=(9.0, 9.0))
    plt.tight_layout()    
    plt.show()

    #Step 1 - Calculate correlation
    # Renewable Energy vs. Co2 emission per capita splits has a low correlation. 
    #The scatter plot confirms that this is a good choice. Picking that combination.     
    print(df_2019_energy_co2.corr())
    
    #Step 2 - Calculate the shillhout score
    print("n score")
    
    for ncluster in range(2, 10):
        
        kmeans = cluster.KMeans(n_clusters=ncluster)
        kmeans.fit(df_2019_energy_co2[['Renewable_energy','CO2']]) 
        labels = kmeans.labels_
        
        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_
        # calculate the silhoutte score
        print(ncluster, skmet.silhouette_score(df_2019_energy_co2[['Renewable_energy','CO2']], labels))
    
    #Number of Clusters - 2
    
    #Step 3 - Normalize
    df_2019_energy_co2_normalized = df_2019_energy_co2[["Renewable_energy", "CO2"]].copy()
    df_2019_energy_co2_normalized, df_min, df_max = ct.scaler(df_2019_energy_co2_normalized)
    
    #Step 4 - K-means Clustering
    ncluster = 2
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_2019_energy_co2_normalized) 
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_2019_energy_co2_normalized["Renewable_energy"], df_2019_energy_co2_normalized["CO2"], 10, labels,marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("Renewable_energy")
    plt.ylabel("CO2 Emission per capita")
    plt.title("Clustering - Renewable Energy vs Co2 emission per capita- 1990")
    plt.show()
    
    # move the cluster centres to the original scale
    nc = 2 # number of cluster centres
    
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(df_2019_energy_co2_normalized) 
    
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    
    plt.figure(figsize=(6.0, 6.0))
    
    # Use the original dataframe
    plt.scatter(df_2019_energy_co2["Renewable_energy"], df_2019_energy_co2["CO2"], c=labels, cmap="tab10")
    # colour map Accent selected to increase contrast between colours
    
    # Step 5 -Rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:,0]
    yc = scen[:,1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    
    plt.xlabel("Renewable energy consumption (% of total final energy consumption)")
    plt.ylabel("CO2 emission per capita")
    plt.title("Co2 emission vs Renewable energy consumption - original scale")
    plt.show()
    


def clustering_Energy_CO2_1990(df_energy,df_gdp,df_co2):
    
    #Choose 1990 column of each dataframe
    df_energy = df_energy[df_energy["1990"].notna()]
    df_co2 = df_co2[df_co2["1990"].notna()]
    
    #Choose only essential columns and take a copy
    df_energy_1990 = df_energy[["Country Name", "1990"]].copy()
    df_co2_1990 = df_co2[["Country Name", "1990"]].copy()
    
    #Merge renewable energy and Co2 emission columns on Country name
    df_1990_energy_co2 = pd.merge(df_energy_1990, df_co2_1990, on="Country Name", how="outer")
    
    #Drop NaNs in the merged dataframe
    df_1990_energy_co2 = df_1990_energy_co2.dropna() 
    
    #Rename the axis
    df_1990_energy_co2 = df_1990_energy_co2.rename(columns={"1990_x":"Renewable_energy", "1990_y":"CO2"})
    
    # scatter plot
    pd.plotting.scatter_matrix(df_1990_energy_co2, figsize=(9.0, 9.0))
    plt.tight_layout()    
    plt.show()
    
    #Step 1 - Calculate correlation
    # Renewable Energy vs. Co2 emission per capita splits has a low correlation. 
    print(df_1990_energy_co2.corr())
    
    #Step 2 - Calculate the shillhout score
    print("n score")

    for ncluster in range(2, 10):
        kmeans = cluster.KMeans(n_clusters=ncluster)
        
        # Fit the data, results are stored in the kmeans object
        kmeans.fit(df_1990_energy_co2[['Renewable_energy','CO2']]) # fit done on x,y pairs
        labels = kmeans.labels_
        
        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_
        
        # calculate the silhoutte score
        print(ncluster, skmet.silhouette_score(df_1990_energy_co2[['Renewable_energy','CO2']], labels))

    #Step 3 - Normalize
    
    df_1990_energy_co2_normalized = df_1990_energy_co2[["Renewable_energy", "CO2"]].copy()
    df_1990_energy_co2_normalized, df_min, df_max = ct.scaler(df_1990_energy_co2_normalized)
    
    #Step 4 - K-means Clustering
    
    #Number of clusters are = 2
    ncluster = 2
    
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_1990_energy_co2_normalized) 
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_1990_energy_co2_normalized["Renewable_energy"], df_1990_energy_co2_normalized["CO2"], 10, labels,marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("Renewable_energy")
    plt.ylabel("CO2 emission per capita")
    plt.title("Clustering - Renewable Energy vs Co2 emission per capita- 2019")
    plt.show()
    
    #Step 5 - move to original scale
    
    #Back to normal
    nc = 2 #number of cluster centres
    
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(df_1990_energy_co2_normalized) 
    
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    
    plt.figure(figsize=(6.0, 6.0))

    # Use the original dataframe
    plt.scatter(df_1990_energy_co2["Renewable_energy"], df_1990_energy_co2["CO2"], c=labels, cmap="tab10")
    
    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:,0]
    yc = scen[:,1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    
    plt.xlabel("Renewable Energy")
    plt.ylabel("CO2 Emission per capita")
    plt.title("Renewable Energy vs Co2 emission per capita - original scale ")
    plt.show()
    

def clustering_GDP_CO2_2019(df_energy,df_gdp,df_co2):
    
    #Choose 1990 column of each dataframe
    df_gdp = df_gdp[df_gdp["2019"].notna()]
    df_co2 = df_co2[df_co2["2019"].notna()]
    
    #Choose only essential columns and take a copy
    df_gdp_2019 = df_gdp[["Country Name", "2019"]].copy()
    df_co2_2019 = df_co2[["Country Name", "2019"]].copy()
    
    #Merge gdp and co2 emission per capita
    df_2019_gdp_co2 = pd.merge(df_gdp_2019, df_co2_2019, on="Country Name", how="outer")

    #Drop NaNs in the dataframe and rename axis
    df_2019_gdp_co2 = df_2019_gdp_co2.dropna() 
    df_2019_gdp_co2 = df_2019_gdp_co2.rename(columns={"2019_x":"GDP per capita", "2019_y":"CO2"})
    
    # scatter plot
    pd.plotting.scatter_matrix(df_2019_gdp_co2, figsize=(9.0, 9.0))
    plt.tight_layout()    
    plt.show()
    
    #Step 1 - Calculate the correlation
    #Correlation is - 0.5
    print(df_2019_gdp_co2.corr())
    
    #Step 2 - Calculate the shillhout score
    print("n score")

    for ncluster in range(2, 10):
        kmeans = cluster.KMeans(n_clusters=ncluster)
        # Fit the data, results are stored in the kmeans object
        kmeans.fit(df_2019_gdp_co2[['GDP per capita','CO2']]) 
        
        labels = kmeans.labels_
        
        # extract the estimated cluster centres
        cen = kmeans.cluster_centers_
        
        # calculate the silhoutte score
        print(ncluster, skmet.silhouette_score(df_2019_gdp_co2[['GDP per capita','CO2']], labels))
    
    #Step 3 - Normalize
    
        df_2019_gdp_co2_normalized = df_2019_gdp_co2[["GDP per capita", "CO2"]].copy()
        df_2019_gdp_co2_normalized, df_min, df_max = ct.scaler(df_2019_gdp_co2_normalized)
        
    #Step 4 - K-means clustering
    #Number of clusters are = 3
    ncluster = 3
    kmeans = cluster.KMeans(n_clusters=ncluster)
    kmeans.fit(df_2019_gdp_co2_normalized) 
    
    labels= kmeans.labels_
    cen = kmeans.cluster_centers_
    
    xcen = cen[:, 0]
    ycen = cen[:, 1]
    
    # cluster by cluster
    plt.figure(figsize=(8.0, 8.0))
    cm = plt.cm.get_cmap('tab10')
    plt.scatter(df_2019_gdp_co2_normalized["GDP per capita"], df_2019_gdp_co2_normalized["CO2"], 10, labels,marker="o", cmap=cm)
    plt.scatter(xcen, ycen, 45, "k", marker="d")
    plt.xlabel("GDP per capita")
    plt.ylabel("CO2 emission per capita")
    plt.title("Clustering - GDP vs Co2 emission - 2019")
    plt.show()

    
    # Step 5 -Rescale and show cluster centres

    nc = 3 # number of cluster centres
    kmeans = cluster.KMeans(n_clusters=nc)
    kmeans.fit(df_2019_gdp_co2_normalized) 
    
    # extract labels and cluster centres
    labels = kmeans.labels_
    cen = kmeans.cluster_centers_
    
    plt.figure(figsize=(6.0, 6.0))
    
    # Use the original dataframe
    plt.scatter(df_2019_gdp_co2["GDP per capita"], df_2019_gdp_co2["CO2"], c=labels, cmap="tab10")
    # colour map Accent selected to increase contrast between colours
    
    # rescale and show cluster centres
    scen = ct.backscale(cen, df_min, df_max)
    xc = scen[:,0]
    yc = scen[:,1]
    plt.scatter(xc, yc, c="k", marker="d", s=80)
    
    plt.xlabel("GDP per capita")
    plt.ylabel("CO2 Emission per capita")
    plt.title("GDP vs CO2 emission - original scale")
    plt.show()
    

    
def logistic(t, n0, g, t0):
    """Calculates the logistic function with scale factor n0 and growth rate g"""
    f = n0 / (1 + np.exp(-g*(t - t0)))
    
    return f



def GDP_logistic_growth():
    #Fitting GDP for Sweden
  
    #Extract GDP data of sweden
    df_gdp_trans=df_gdp.transpose()
    df_gdp_trans.columns = df_gdp_trans.iloc[0]
    df_gdp_trans = df_gdp_trans.drop(df_gdp_trans.index[[0,1]])
    df_gdp_swd = df_gdp_trans[['Sweden']]
    df_gdp_swd = df_gdp_swd.dropna()
    df_gdp_swd = df_gdp_swd.reset_index()
    df_gdp_swd = df_gdp_swd.rename(columns={'index':'Year','Sweden':'GDP'})
    df_gdp_swd['Year'] = df_gdp_swd['Year'].astype(int)
    df_gdp_swd['GDP'] = df_gdp_swd['GDP'].astype(float)

    # Extract the year and total GDP columns as numpy arrays
    x = df_gdp_swd['Year']
    y = df_gdp_swd['GDP'].values
    
    # Define the initial guess for the parameters
    p0 = [max(y), 1, np.median(x)]
    
    # Fit the logistic model to the data
    popt, pcorr = opt.curve_fit(logistic, x, y, p0)
    
    print("Fit parameter", popt)
    
    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(pcorr))
    
    df_gdp_swd["pop_logistics"] = logistic(df_gdp_swd["Year"], *popt)
    
    # call function to calculate upper and lower limits with extrapolation
    # create extended year range
    years = np.arange(1960, 2040)
    lower, upper = err.err_ranges(years, logistic, popt, sigmas)
    
    plt.figure()
    plt.title("Fitting - GDP per capita and Predictions")
    plt.xlabel("Years")
    plt.ylabel("GDP per capita (current US$)")
    plt.plot(df_gdp_swd["Year"], df_gdp_swd["GDP"], label="data")
    plt.plot(df_gdp_swd["Year"], df_gdp_swd["pop_logistics"], label="fit")
    # plot error ranges with transparency
    plt.fill_between(years, lower, upper, alpha=0.5)
    
    plt.legend(loc="upper left")
    plt.show()
    
    #Predict values for 20 years
    print("GDP in")
    print("2030:", logistic(2030, *popt))
    print("2040:", logistic(2040, *popt))
    print("2050:", logistic(2050, *popt))
    
    lower, upper = err.err_ranges(2030, logistic, popt, sigmas)
    print("GDP in")
    print("2030: between", round(lower), "and", round(upper))
    
    
    # calculate sigma of prediction from ranges
    sigma = (upper-lower) / 2.0
    print()
    print("GDP in")
    print("2030", round(logistic(2030, *popt), 0), "+/-", round(sigma, 0))
    
    

def Co2_logistic_growth():
    df_co2_trans=df_co2.transpose()
    df_co2_trans.columns = df_co2_trans.iloc[0]
    df_co2_trans = df_co2_trans.drop(df_co2_trans.index[[0,1]])
    df_co2_swd = df_co2_trans[['Sweden']]
    df_co2_swd = df_co2_swd.dropna()
    df_co2_swd = df_co2_swd.reset_index()
    df_co2_swd = df_co2_swd.rename(columns={'index':'Year','Sweden':'Co2'})
    df_co2_swd['Year'] = df_co2_swd['Year'].astype(int)
    df_co2_swd['Co2'] = df_co2_swd['Co2'].astype(float)

    # Extract the year and total GDP columns as numpy arrays
    x = df_co2_swd['Year']
    y = df_co2_swd['Co2'].values
    
    # Define the initial guess for the parameters
    p0 = [max(y), 1, np.median(x)]
    
    # Fit the logistic model to the data
    popt, pcorr = opt.curve_fit(logistic, x, y, p0)
    
    print("Fit parameter", popt)
    
    # extract variances and calculate sigmas
    sigmas = np.sqrt(np.diag(pcorr))
    
    df_co2_swd["pop_logistics"] = logistic(df_co2_swd["Year"], *popt)
    
    # call function to calculate upper and lower limits with extrapolation
    # create extended year range
    years = np.arange(1990, 2040)
    lower, upper = err.err_ranges(years, logistic, popt, sigmas)
    
    plt.figure()
    plt.title("fitting - Co2 emission per capita and Predictions")
    plt.xlabel("Years")
    plt.ylabel("Co2 emission per capita")
    plt.plot(df_co2_swd["Year"], df_co2_swd["Co2"], label="data")
    plt.plot(df_co2_swd["Year"], df_co2_swd["pop_logistics"], label="fit")
    # plot error ranges with transparency
    plt.fill_between(years, lower, upper, alpha=0.5)
    plt.legend(loc="upper left")
    plt.show()             

    #Predict values for 20 years
    print("CO2 emission in")
    print("2030:", logistic(2030, *popt))
    print("2040:", logistic(2040, *popt))
    print("2050:", logistic(2050, *popt))
    
    lower, upper = err.err_ranges(2030, logistic, popt, sigmas)
    print("Co2 in")
    print("2030: between", round(lower), "and", round(upper), "Mill")
    
    # calculate sigma of prediction from ranges
    sigma = (upper-lower) / 2.0
    print()
    print("Co2 in")
    print("2030", round(logistic(2030, *popt), 0), "+/-", round(sigma, 0),"Mill.")
        







df_energy,df_gdp,df_co2=read_file("Renewable_energy.csv","Gdp_percapita.csv","Co2_emission.csv")
clustering_Energy_CO2_2019(df_energy,df_gdp,df_co2)
clustering_Energy_CO2_1990(df_energy,df_gdp,df_co2)
clustering_GDP_CO2_2019(df_energy,df_gdp,df_co2)
GDP_logistic_growth()
Co2_logistic_growth()