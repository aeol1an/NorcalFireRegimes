from datetime import datetime
from pathlib import Path
root = Path().cwd()

import numpy as np
import xarray as xr

from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

import coordinate_utils as cu
import vec_transform as vt
import trend_utils as tu

#pre-training
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#models
from sklearn.cluster import KMeans

#metrics
from sklearn.metrics import silhouette_samples, davies_bouldin_score

#Parameters
frp_percent_thresholds = np.linspace(.9, .99, 10)

for frp_percent_threshold in frp_percent_thresholds:
    #Output Dirs
    imgdir = root/"img"/"feature_eng"/str(frp_percent_threshold)
    imgdir.mkdir(exist_ok=True)
    dbimgdir = (imgdir/"davies_bouldin")
    silhimgdir = (imgdir/"silhouette")
    dbimgdir.mkdir(exist_ok=True)
    silhimgdir.mkdir(exist_ok=True)

    #Domains
    norcal_domain = [-124.65, -117.25, 35.35, 42.5]
    synoptic_domain = [-150, -90, 20, 60]

    frp = xr.open_mfdataset(str(root/"nc"/"frp"/"frp*.nc"), engine="netcdf4")
    norcal_frp = frp.sel(longitude=slice(cu.convert_lon_0_360(norcal_domain[0]),cu.convert_lon_0_360(norcal_domain[1])))
    norcal_frp = norcal_frp.sel(latitude=slice(norcal_domain[3],norcal_domain[2]))["frpfire"]
    norcal_frp = norcal_frp.sel(valid_time=slice(cu.ts_to_dt('2003-01-02T00'), cu.ts_to_dt('2024-11-10T00')))
    norcal_frp_np = norcal_frp.values

    lonlon_frp, latlat_frp = np.meshgrid(norcal_frp.coords['longitude'].values, norcal_frp['latitude'].values)
    sa_frp = cu.sphere_surface_area(latlat_frp, lonlon_frp, 0.1)
    avgnorcal_frp_np = np.sum(norcal_frp_np * sa_frp, axis=(1,2))/np.sum(sa_frp, axis=(0,1))


    frp_value_threshold = np.sort(avgnorcal_frp_np)[np.floor(avgnorcal_frp_np.size*frp_percent_threshold).astype(np.int32):][0]
    bounded_frp = norcal_frp[avgnorcal_frp_np > frp_value_threshold]

    fullnames = {
        "z": "500hPa Geopotential Height",
        "t2m": "2m Air Temperature",
        "d2m": "2m Dew Point Temperature",
        "msl": "Mean Sea Level Pressure"
    }

    from itertools import combinations
    allvars = ['z', "t2m", "d2m", "msl"]
    def get_all_subsets(lst):
        subsets = []
        for i in range(len(lst) + 1):
            subsets.extend(list(combinations(lst, i)))
        return subsets
    combs = [list(a) for a in get_all_subsets(allvars)]
    combs.pop(len(combs)-1)
    print("Combinations:", combs)

    for comb in combs:
        vars = [x for x in allvars if x not in comb]
        curr_fullnames = ""
        for var in vars:
            curr_fullnames += fullnames[var] + ", "
        curr_fullnames = curr_fullnames[:-2]
        
        era5 = xr.open_mfdataset(str(root/"nc"/"era5"/"*.nc"), join="inner")
        era5['z'] = era5['z'][:,0,:,:]

        era5 = era5.drop_dims(['pressure_level'])
        era5 = era5.drop_vars(['number', 'expver'])
        era5 = era5.drop_vars(['u10', 'v10'])

        print("Dropped vars:", comb)
        print("Kept vars:", curr_fullnames)
        era5 = era5.drop_vars(comb)
        era5_scaled_and_detrended = era5.copy()
        
        lonlon_syn, latlat_syn = np.meshgrid(era5.coords['longitude'].values, era5.coords['latitude'].values)
        sa_syn = cu.sphere_surface_area(latlat_syn, lonlon_syn, 0.25)
        area_weight = sa_syn / np.mean(sa_syn, axis=(0,1))
        
        print("Starting Detrending and Weighting:")
        for var_name in era5.data_vars:
            values = era5[var_name].values
            weighted_values = values*area_weight
            long_term_detrended_weighted_values, _ = \
                tu.quadratic_detrend(weighted_values)
            long_term_seasonal_detrended_weighted_values, _ = \
                tu.seasonal_detrend(long_term_detrended_weighted_values)
            era5_scaled_and_detrended[var_name].values = long_term_seasonal_detrended_weighted_values
            
        era5_frp_subset = era5.sel(valid_time=bounded_frp.coords['valid_time'].values)
        era5_sad_frp_subset = era5_scaled_and_detrended.sel(valid_time=bounded_frp.coords['valid_time'].values)
        
        X, shapes = vt.fields_to_rows(tuple(era5_sad_frp_subset[var_name].values for var_name in era5_sad_frp_subset.data_vars))
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        print("Starting PCA:")
        pca = PCA(n_components=0.95, svd_solver='full')
        X_pca = pca.fit_transform(X_scaled)
        print(f"Reduced from {X.shape[1]} to {X_pca.shape[1]} features")
        
        k_init = 'k-means++'
        k_iter = 1000
        k_runs = 10
        
        kmeans_errors_davies_bouldin = xr.DataArray(
            np.empty((14, 25)),
            dims=["k", "n"],
            coords={
                "k": np.arange(2, 16),
                "n": np.arange(25)
            }
        )

        print("Starting Davies_Bouldin:")
        for k in kmeans_errors_davies_bouldin.coords["k"].values:
            for n in kmeans_errors_davies_bouldin.coords["n"].values:
                km = KMeans(n_clusters=k, init=k_init, n_init=k_runs, max_iter=k_iter)
                km.fit(X_pca)
                kmeans_errors_davies_bouldin.loc[k, n] = km.inertia_
                kmeans_errors_davies_bouldin.loc[k, n] = davies_bouldin_score(X_pca, km.labels_)

        plt.figure(figsize=(11, 5))
        line, = plt.plot(kmeans_errors_davies_bouldin.coords["k"].values, np.mean(kmeans_errors_davies_bouldin, axis=1), marker='.')
        plt.errorbar(line.get_xdata(), line.get_ydata(), yerr=np.std(kmeans_errors_davies_bouldin, axis=1), fmt='o', capsize=5)
        plt.ylabel("Davies-Bouldin Score")
        plt.title("Davies-Bouldin Score and Standard Deviation (25 runs)\nVariables: " + curr_fullnames)
        
        title="dropped"
        for field in comb:
            title += f"_{field}"
            
        plt.savefig(dbimgdir/f"{title}.png", bbox_inches='tight')
        plt.close()
        
        print("Starting Silhouette:")
        (silhimgdir/title).mkdir(exist_ok=True)
        
        
        #-----THIS CODE IS FROM LECTURE (Lecture 7 - Kmeans.ipynb) I DID NOT WRITE IT-------------------
        #I will write comments explaining the code so that I'm not just blatantly copying

        #We need a plot for each value of K, so loop through and create the figure
        for k in kmeans_errors_davies_bouldin.coords["k"].values:
            fig = plt.figure()
            ax1 = plt.axes()
            fig.set_size_inches(11, 7)
            
            # Run the Kmeans algorithm with our hyperparameters and get label to silhouette analysis
            km = KMeans(n_clusters=k, init=k_init, n_init=k_runs, max_iter=k_iter)
            labels = km.fit_predict(X_pca)

            # Get silhouette samples
            silhouette_vals = silhouette_samples(X_pca, labels)

            #Need to loop through each cluster to plot silhouette value for all input vectors
            y_lower, y_upper = 0, 0
            for i, cluster in enumerate(np.unique(labels)):
                #subset to get only the values from our current cluster
                cluster_silhouette_vals = silhouette_vals[labels == cluster]
                cluster_silhouette_vals.sort()
                #adjust beginning of plot where the first bar goes in the cluster
                y_upper += len(cluster_silhouette_vals)
                #plot and add label
                ax1.barh(range(y_lower, y_upper), cluster_silhouette_vals, edgecolor='none', height=1)
                ax1.text(-0.03, (y_lower + y_upper) / 2, str(i + 1))
                #set the lower bound for the next cluster
                y_lower += len(cluster_silhouette_vals)

            # Get the average silhouette score and plot it
            avg_score = np.mean(silhouette_vals)
            ax1.axvline(avg_score, linestyle='--', linewidth=2, color='green')
            ax1.set_yticks([])
            ax1.set_xlim([-0.3, 1])
            ax1.set_xlabel('Silhouette coefficient values')
            ax1.set_ylabel('Cluster labels')
            ax1.set_title(f'Silhouette plot for the various clusters at k={k}\nVariables: ' + curr_fullnames, y=1.02)
            plt.savefig(silhimgdir/title/f"{k}.png", bbox_inches='tight')
            plt.close()
            #-----THIS CODE IS FROM LECTURE (Lecture 7 - Kmeans.ipynb) I DID NOT WRITE IT---------------