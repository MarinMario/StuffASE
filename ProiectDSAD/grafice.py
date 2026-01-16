import numpy as np
import pandas as pd
from matplotlib import cm, pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from seaborn import scatterplot, kdeplot


def plot_ierarhie(h:np.ndarray,etichete=None,color_threshold=0,
                  titlu="Plot ierarhie"):
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16})
    dendrogram(h,labels=etichete,color_threshold=color_threshold,ax=ax)

def show():
    plt.show()

def plot_partitie(
        t_z:pd.DataFrame,
        t_gz:pd.DataFrame,
        clase,
        p,
        scor_silh,
        axa_x="Z1",
        axa_y="Z2",
        titlu="Plot partitie in axele principale",
        etichete=True
):
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(1,1,1,aspect=1)
    ax.set_title(titlu+" Scor Silhouette "+str(scor_silh),
                 fontdict={"fontsize":16})
    scatterplot(t_z, x=axa_x, y=axa_y, hue=p,
                hue_order=clase, ax=ax)
    scatterplot(t_gz,x=axa_x,y=axa_y,
                hue=clase,hue_order=clase,legend=False,
                marker = "s",s=100,ax=ax
                )
    if etichete:
        n = len(t_z)
        for i in range(n):
            ax.text(t_z[axa_x].iloc[i],t_z[axa_y].iloc[i],t_z.index[i])

def f_distributie(t_z:pd.DataFrame, p, clase, variabila,
                  titlu="Plot distributii pe clase"):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu+" Variabila:"+variabila, fontdict={"fontsize": 16})
    kdeplot(t_z, x=variabila, hue=p, hue_order=clase,
            ax=ax, warn_singular=False, fill=True)

def histograme(t:pd.DataFrame,variabila,p,clase,titlu="Plot histograme"):
    fig = plt.figure(figsize=(12, 7))
    q = len(clase)
    ax = fig.subplots(1,q,sharey=True)
    fig.suptitle(titlu+" Variabila:"+variabila, fontdict={"fontsize": 16})
    x = t[variabila].values
    for i in range(q):
        axa = ax[i]
        y = x[p==clase[i]]
        assert isinstance(axa,plt.Axes)
        axa.hist(y,10,rwidth=0.9,range=(min(x),max(x)))
        axa.set_xlabel(clase[i])

def plot_silhouette(silh_values, p, titlu="Grafic Silhouette"):
    """
    silh_values: ndarray - scorurile individuale (silhouette_samples) deja calculate
    p: ndarray - vectorul cu etichetele clusterelor (C1, C2...)
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    clase = np.unique(p)
    n_clusters = len(clase)
    avg_score = np.mean(silh_values) # Singurul calcul simplu: media
    
    y_lower = 10
    for i, cluster in enumerate(clase):
        # Doar extragem valorile gata calculate pentru clusterul curent
        ith_cluster_silh_vals = silh_values[p == cluster]
        ith_cluster_silh_vals.sort()
        
        size_cluster_i = ith_cluster_silh_vals.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = cm.nipy_spectral(float(i) / n_clusters)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silh_vals,
                          facecolor=color, edgecolor=color, alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))
        y_lower = y_upper + 10

    ax.set_title(f"{titlu}\nScor mediu: {round(avg_score, 3)}")
    ax.axvline(x=avg_score, color="red", linestyle="--")
    ax.set_yticks([]) 
    ax.set_xlabel("Coeficient Silhouette")