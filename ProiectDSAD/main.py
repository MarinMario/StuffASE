import pandas as pd
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.cluster.hierarchy import linkage
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from functii import calcul_partitie, salvare_ndarray
from grafice import f_distributie, histograme, plot_ierarhie, plot_partitie, plot_silhouette, show

def incarca_date():
  date = pd.read_csv("date_in/date.csv", index_col=0)
  return date

def standardizeaza_date(date):
  scaler = StandardScaler()
  standardized_data = scaler.fit_transform(date)
  standardized_df = pd.DataFrame(standardized_data, columns=date.columns, index=date.index)
  standardized_df.to_csv("date_out/date_standardizate.csv")
  return standardized_df

def clusterizare(date_numerice):
  ierarhie = linkage(date_numerice, method='ward')
  return ierarhie

def analiza_partitie(date, ierarhie, nume_partitie, nr_clusteri=None):
  nr_clustere, color_threshold, componente_partitie = calcul_partitie(ierarhie, nr_clusteri)
  scoruri_silhouette_individuale = silhouette_samples(date.values, componente_partitie)
  scor_silhouette = silhouette_score(date.values, componente_partitie)
  plot_ierarhie(ierarhie, date.index, color_threshold, f"Plot ierarhie {nume_partitie}")
  print(f"Numar clusteri {nume_partitie}:", nr_clustere)
  print(f"Distanta prag {nume_partitie}:", color_threshold)
  plot_silhouette(scoruri_silhouette_individuale, componente_partitie, f"Analiza Silhouette - {nume_partitie}")
  return {
    "nr_clustere": nr_clustere,
    "color_threshold": color_threshold,
    "componente_partitie": componente_partitie,
    "scor_silhouette": scor_silhouette,
    "scoruri_silhouette_individuale": scoruri_silhouette_individuale
  }

def plot_partitie_in_axele_principale(date, componente_partitie, scor_silhouette, nume_partitie):
  acp = PCA(n_components=2)
  z = acp.fit_transform(date.values)
  t_z = salvare_ndarray(
      z,
      date.index,
      ["Z1","Z2"],
      date.index.name
  )
  t_gz = t_z.groupby(by=componente_partitie).mean()
  plot_partitie(
      t_z,
      t_gz,
      np.unique(componente_partitie),
      componente_partitie,
      scor_silhouette,
      titlu=nume_partitie
  )

def analiza_partitie_prin_grafice_de_distributie(variabile_observate, date, componente_partitie, nume_partitie):
  # Analiza partitie prin grafice de distributie
  for variabila in variabile_observate:
      f_distributie(
          date,
          componente_partitie,
          np.unique(componente_partitie),
          variabila,
          titlu="Distributie pe clase - " + nume_partitie + " - "
      )
      histograme(
          date,
          variabila,
          componente_partitie,
          np.unique(componente_partitie),
          titlu="Histograme pe clase - " + nume_partitie + " - " 
      )

def plot_grafice_finale_partitie(date_standardizate, partitie, nume_partitie):
  plot_partitie_in_axele_principale(date_standardizate,
                                   partitie["componente_partitie"],
                                   partitie["scor_silhouette"],
                                   nume_partitie)
  analiza_partitie_prin_grafice_de_distributie(date_standardizate.columns, date_standardizate, partitie["componente_partitie"], nume_partitie)

def main():
  date = incarca_date()
  date_standardizate = standardizeaza_date(date)
  ierarhie = clusterizare(date_standardizate.values)

  plot_ierarhie(ierarhie, etichete=date_standardizate.index, titlu="Dendrograma date standardizate")

  partitia_optimala = analiza_partitie(date_standardizate, ierarhie, "Partitia Optimala")
  plot_grafice_finale_partitie(date_standardizate, partitia_optimala, "Partitia Optimala")

  partitia_cu_3_clusteri = analiza_partitie(date_standardizate, ierarhie, "Partitia cu 3 Clusteri", 3)
  plot_grafice_finale_partitie(date_standardizate, partitia_cu_3_clusteri, "Partitia cu 3 Clusteri")

  tabel_final = pd.DataFrame(index=date.index)
  tabel_final["Partitie_Optimala"] = partitia_optimala["componente_partitie"]
  tabel_final["Partitie_3_Clustere"] = partitia_cu_3_clusteri["componente_partitie"]
  tabel_final.to_csv("date_out/tabel_partitii.csv")

  show()

main()