#import necessary libraries
import pandas as pd
import numpy as np

import sklearn.cluster as cluster

import matplotlib.pyplot as plt
import cluster_tools as ct

from scipy.optimize import curve_fit
import scipy.optimize as opt
import errors as err
import sklearn.metrics as skmet

#impoert Renewable_energy dataset
df_energy = pd.read_csv("Renewable_energy.csv",skiprows=4)
df_energy = df_energy.drop(['Country Code', 'Indicator Code'], axis=1)
df_energy.describe()

df_gdp = pd.read_csv("Gdp_percapita.csv",skiprows=4)
df_gdp = df_gdp.drop(['Country Code', 'Indicator Code'], axis=1)

#Import CO2 emission dataset
df_co2 = pd.read_csv("Co2_emission.csv",skiprows=4)
df_co2 = df_co2.drop(['Country Code', 'Indicator Code'], axis=1)
df_co2.describe()

#Choose 2019 column of each dataframe
df_energy = df_energy[df_energy["2019"].notna()]
df_gdp = df_gdp[df_gdp["2019"].notna()]
df_co2 = df_co2[df_co2["2019"].notna()]

#Choose only essential columns and take a copy
df_energy_2019 = df_energy[["Country Name", "2019"]].copy()
df_gdp_2019 = df_gdp[["Country Name", "2019"]].copy()
df_co2_2019 = df_co2[["Country Name", "2019"]].copy()

#Merge renewable energy and Co2 emission columns on Country name
df_2019_energy_co2 = pd.merge(df_energy_2019, df_co2_2019, on="Country Name", how="outer")
df_2019_energy_co2.to_excel("energy_co2.xlsx")

#Rename axis
df_2019_energy_co2 = df_2019_energy_co2.dropna() 
df_2019_energy_co2 = df_2019_energy_co2.rename(columns={"2019_x":"Renewable_energy", "2019_y":"CO2"})

# heatmap
ct.map_corr(df_2019_energy_co2, 4)

# scatter plot
pd.plotting.scatter_matrix(df_2019_energy_co2, figsize=(9.0, 9.0))
plt.tight_layout()    
plt.show()


#Merge gdp and co2 emission
df_2019_gdp_co2 = pd.merge(df_gdp_2019, df_co2_2019, on="Country Name", how="outer")


df_2019_gdp_co2 = df_2019_gdp_co2.dropna() 
df_2019_gdp_co2 = df_2019_gdp_co2.rename(columns={"2019_x":"GDP per capita", "2019_y":"CO2"})


# heatmap
ct.map_corr(df_2019_gdp_co2, 4)

# scatter plot
pd.plotting.scatter_matrix(df_2019_gdp_co2, figsize=(9.0, 9.0))
plt.tight_layout()    
plt.show()


#Elbow method
k_values = [1,2,3,4,5,6,7,8,9,10]
wcss_error = []
for k in k_values:
    model = cluster.KMeans(n_clusters=k)
    model.fit(df_2019_gdp_co2[['GDP per capita','CO2']]) 
    wcss_error.append(model.inertia_)

plt.plot(k_values,wcss_error)


print("n score")
# loop over number of clusters
for ncluster in range(2, 10):
    # set up the clusterer with the number of expected clusters
    kmeans = cluster.KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(df_2019_gdp_co2[['GDP per capita','CO2']]) # fit done on x,y pairs
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(df_2019_gdp_co2[['GDP per capita','CO2']], labels))
    
df_2019_gdp_co2_normalized = df_2019_gdp_co2[["GDP per capita", "CO2"]].copy()
# normalise
df_2019_gdp_co2_normalized, df_min, df_max = ct.scaler(df_2019_gdp_co2_normalized)

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
plt.ylabel("CO2")
plt.title("Clustering - GDP vs Co2 emission")
plt.show()

nc = 3 # number of cluster centres

kmeans = cluster.KMeans(n_clusters=nc)
kmeans.fit(df_2019_gdp_co2_normalized) 
# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(6.0, 6.0))
# scatter plot with colours selected using the cluster numbers
# now using the original dataframe
plt.scatter(df_2019_gdp_co2["GDP per capita"], df_2019_gdp_co2["CO2"], c=labels, cmap="tab10")
# colour map Accent selected to increase contrast between colours

# rescale and show cluster centres
scen = ct.backscale(cen, df_min, df_max)
xc = scen[:,0]
yc = scen[:,1]
plt.scatter(xc, yc, c="k", marker="d", s=80)

plt.xlabel("GDP per capita")
plt.ylabel("CO2")
plt.title("3 clusters")
plt.show()