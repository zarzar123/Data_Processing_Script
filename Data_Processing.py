#!/usr/bin/env python
# coding: utf-8

# In[8]:


pip install pandas numpy matplotlib scikit-learn geopandas shapely folium fpdf


# In[13]:


pip install selenium


# In[17]:


pip install chromedriver


# In[24]:


# Import necessary libraries
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from shapely.geometry import Point
from sklearn.metrics import silhouette_score

# Step 1: Generate Synthetic Geospatial Data
np.random.seed(42)
n_samples = 200
center_lat, center_lon = 45.4215, -75.6972  # Approximate center (Ottawa, Canada)
latitudes = np.random.normal(center_lat, 0.1, n_samples)
longitudes = np.random.normal(center_lon, 0.1, n_samples)

# Create a DataFrame with geospatial data
data = pd.DataFrame({'Latitude': latitudes, 'Longitude': longitudes})

# Convert to GeoDataFrame
geometry = [Point(xy) for xy in zip(data['Longitude'], data['Latitude'])]
gdf = gpd.GeoDataFrame(data, geometry=geometry)

# Display the first few rows of the GeoDataFrame
print("First 5 rows of the geospatial data:")
display(gdf.head())

# Step 2: Data Visualization - Initial Geospatial Data
print("\nInitial Geospatial Data Visualization:")
gdf.plot(marker='o', color='blue', markersize=5, figsize=(8, 6))
plt.title("Geospatial Data Points")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Step 3: Data Preprocessing - Standardize Geospatial Features
scaler = StandardScaler()
X = scaler.fit_transform(gdf[['Longitude', 'Latitude']])

# Step 4: Clustering - K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
gdf['kmeans_labels'] = kmeans.fit_predict(X)

# Clustering - DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
gdf['dbscan_labels'] = dbscan.fit_predict(X)

# Evaluate K-Means Clustering with Silhouette Score
kmeans_silhouette = silhouette_score(X, gdf['kmeans_labels'])
print(f"\nSilhouette Score for K-Means: {kmeans_silhouette:.2f}")

# Step 5: Visualize K-Means Clustering with Static Map
print("\nK-Means Clustering Visualization (Static Map):")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
gdf.plot(ax=ax, column='kmeans_labels', categorical=True, legend=True, cmap='Set1', markersize=5)
plt.title("K-Means Clustering on Geospatial Data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Step 6: Interactive Map Visualization using Folium
print("\nInteractive Map Visualization with Folium:")
# Initialize a folium map centered around the data points
m = folium.Map(location=[center_lat, center_lon], zoom_start=12)

# Add clustered points to the map
for idx, row in gdf.iterrows():
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=5,
        color='red' if row['kmeans_labels'] == 0 else 'blue' if row['kmeans_labels'] == 1 else 'green' if row['kmeans_labels'] == 2 else 'purple',
        fill=True,
        fill_opacity=0.6,
        popup=f"Cluster: {row['kmeans_labels']}"
    ).add_to(m)

# Save the Folium map as an HTML file
map_file = "kmeans_clustering_map.html"
m.save(map_file)
print(f"\nInteractive map saved as {map_file}")

# Step 7: Visualize DBSCAN Clustering with Static Map
print("\nDBSCAN Clustering Visualization (Static Map):")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
gdf.plot(ax=ax, column='dbscan_labels', categorical=True, legend=True, cmap='tab10', markersize=5)
plt.title("DBSCAN Clustering on Geospatial Data")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.show()

# Step 8: Calculate Centroids for K-Means Clusters
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=['Longitude', 'Latitude'])
print("\nCentroids of K-Means Clusters:")
display(centroids)

# Convert centroids to GeoDataFrame
centroid_geometry = [Point(xy) for xy in zip(centroids['Longitude'], centroids['Latitude'])]
centroid_gdf = gpd.GeoDataFrame(centroids, geometry=centroid_geometry)

# Step 9: Plot Centroids on the K-Means Map
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
gdf.plot(ax=ax, column='kmeans_labels', categorical=True, legend=True, cmap='Set1', markersize=5)
centroid_gdf.plot(ax=ax, marker='x', color='black', markersize=100, label='Centroids')
plt.title("K-Means Clustering with Centroids")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.legend()
plt.show()

# Step 10: Add Centroids to the Folium Map and Save as HTML
print("\nInteractive Map with Centroids (Folium):")
# Add the centroids to the folium map
for _, row in centroid_gdf.iterrows():
    folium.Marker(
        location=(row['Latitude'], row['Longitude']),
        popup="Centroid",
        icon=folium.Icon(color='black', icon='info-sign')
    ).add_to(m)

# Save the updated Folium map as an HTML file
updated_map_file = "kmeans_clustering_with_centroids_map.html"
m.save(updated_map_file)
print(f"\nUpdated interactive map saved as {updated_map_file}")


# In[25]:


import os
print(os.getcwd())


# In[ ]:




