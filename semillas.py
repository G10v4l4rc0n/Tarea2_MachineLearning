import re
import pandas as pd

######IMPORTACIÓN Y TRANSFORMACIÓN DE DATASET A DATAFRAME
#IMPORTACIÓN
with open("seeds_dataset.txt", 'r') as file:
    data_list = re.split('\t|\n', file.read())

#TRANSFORMACIÓN A LISTA
copy = data_list.copy()
for value in copy:
    try:
        index_to_remove = copy.index('')
    except:
        break
    else:
        copy.remove('')

for index in range(len(copy)):
    copy[index] = float(copy[index])

#TRANSFORMACIÓN A DATAFRAME
final_list, sub_list = [], []
cont = 0;
for i in range(210):
    final_list.append(copy[8*i:8*(i+1)])

#Revisar si la nueva lista tiene los valores correctos
"""
for val in final_list[:5]:
    print(val)
"""

data ={
    "area": [],
    "perimeter": [],
    "compactness": [],
    "length of kernel": [],
    "width of kernel": [],
    "asymmetry coefficient": [],
    "length of kernel groove": [],
    "classification": []
}
for sub_list in final_list:
    data["area"].append(sub_list[0])
    data["perimeter"].append(sub_list[1])
    data["compactness"].append(sub_list[2])
    data["length of kernel"].append(sub_list[3])
    data["width of kernel"].append(sub_list[4])
    data["asymmetry coefficient"].append(sub_list[5])
    data["length of kernel groove"].append(sub_list[6])
    data["classification"].append(sub_list[7])
df = pd.DataFrame(data)
#print(df)

df_dtypes = {
    "area": float,
    "perimeter": float,
    "compactness": float,
    "length of kernel": float,
    "width of kernel": float,
    "asymmetry coefficient": float,
    "length of kernel groove": float,
    "classification": int
}

df = df.astype(df_dtypes)
#print(df.dtypes)
#print(df.astype(df_dtypes).dtypes)
#input("ctrl + c para terminar ejecución")

######CLUSTERIZACIÓN INICIAL SIN PREPROCESAMIENTO

##importación de librerías

#Generales
import numpy as np
import matplotlib.pyplot as plt

#plt.scatter(df.area, df.classification)
#plt.show()

#Modelos y metricas
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import pair_confusion_matrix, contingency_matrix

#mezclar dataframe
df = df.sample(frac=1, random_state=42)
#print(df)

scoring = [
    "adjusted_mutual_info_score",
    "adjusted_rand_score",
    "homogeneity_score"
]

#KMEANS
param_grid = {
    'n_clusters': [1, 2, 3, 4, 5, 10, 15, 20],
    'init': ['k-means++', 'random'],
    'n_init': ['auto', 1, 10, 100],
    'max_iter': [10, 100, 1000],
    'random_state': [42]
}

kmeans = KMeans()
grid_kmeans = GridSearchCV(estimator=kmeans, param_grid=param_grid, scoring=scoring, refit=False)
print("\n\nIniciando entrenamiento de KMEANS")
grid_kmeans.fit(df)
print("\n\nEntrenamiento terminado")

#DBSCAN
param_grid = [
    {
        'eps': [0.01, 0.1, 0.5, 0.9],
        'min_samples': [1, 5, 10, 15],
        'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
        'algorithm': ['ball_tree', 'kd_tree'],
        'leaf_size': [30, 100, 1000],
        'n_jobs': [2]
    },
    {
        'eps': [0.01, 0.1, 0.5, 0.9],
        'min_samples': [1, 5, 10, 15],
        'metric': ['minkowski'],
        'p': [2, 3, 4, 5],
        'algorithm': ['ball_tree', 'kd_tree'],
        'leaf_size': [30, 100, 1000],
        'n_jobs': [2]
    },
    {
        'eps': [0.01, 0.1, 0.5, 0.9],
        'min_samples': [1, 5, 10, 15],
        'metric': ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan'],
        'algorithm': ['brute'],
        'n_jobs': [2]
    },
    {
        'eps': [0.01, 0.1, 0.5, 0.9],
        'min_samples': [1, 5, 10, 15],
        'metric': ['minkowski'],
        'p': [2, 3, 4, 5],
        'algorithm': ['brute'],
        'n_jobs': [2]
    }
]

dbscan = DBSCAN()
grid_dbscan = GridSearchCV(estimator=dbscan, param_grid=param_grid, scoring=scoring, refit=False)
print("\n\nIniciando entrenamiento de DBSCAN")
grid_dbscan.fit(df)
print("\nEntrenamiento terminado")

#Hierarchical (agglomerative)
param_grid = [
    {
        'n_clusters': [2, 3, 4, 5, 10],
        'metric': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine'],
        'linkage': ['complete', 'average', 'single'],
        'compute_distances': [True]
    },
    {
        'n_clusters': [2, 3, 4, 5, 10],
        'metric': ['euclidean'],
        'linkage': ['ward'],
        'compute_distances': [True]
    },
]

hier = AgglomerativeClustering()
grid_hier = GridSearchCV(estimator=hier, param_grid=param_grid, scoring=scoring, refit=False)
print("\n\nIniciando entrenamiento de HIERARCHICAL")
grid_hier.fit(df)
print("\nEntrenamiento terminado")

print("\n\t\tResultados:")

print("\n\tKMEANS")
print(f"Best estimator: {grid_kmeans.best_estimator_}\n"+
      f"Best score: {grid_kmeans.best_score_}")

print("\n\tDBSCAN")
print(f"Best estimator: {grid_dbscan.best_estimator_}\n"+
      f"Best score: {grid_dbscan.best_score_}")

print("\n\tHIERARCHICAL")
print(f"Best estimator: {grid_hier.best_estimator_}\n"+
      f"Best score: {grid_hier.best_score_}")
"""

print("\n\tKMEANS")
results = pd.DataFrame(grid_kmeans.cv_results_).sort_values(by="rank_test_score")
print(results)
"""

















