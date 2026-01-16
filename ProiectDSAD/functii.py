import pandas as pd
import numpy as np
import numpy as np
import pandas as pd

def calcul_partitie(h: np.ndarray, k=None):
    # Numar jonctiuni:
    m = h.shape[0]
    # Numar instante:
    n = m + 1
    if k is None:
        # Distante dintre jonctiuni:
        d = h[1:, 2] - h[:m - 1, 2]
        # Jonctiunea de diferenta maxima
        j = np.argmax(d) + 1
        k = n - j
    else:
        j = n - k
    color_threshold = (h[j, 2] + h[j - 1, 2]) / 2
    c = np.arange(n)
    for i in range(j):
        k1 = h[i, 0]
        k2 = h[i, 1]
        c[c == k1] = n + i
        c[c == k2] = n + i
    p =np.array(["C"+str(v+1) for v in pd.Categorical(c).codes])
    return k, color_threshold, p

def salvare_ndarray(x:np.ndarray,nume_linii,nume_coloane,nume_index="",nume_fisier_output=None):
    temp = pd.DataFrame(x,nume_linii,nume_coloane)
    temp.index.name=nume_index
    if nume_fisier_output is not None:
        temp.to_csv(nume_fisier_output)
    return temp