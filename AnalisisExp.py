# Importar bibliotecas necesarias
import pandas as pd
import matplotlib.pyplot as plt

# Cargar datos
datos = pd.read_csv("train.csv")  # Asegúrate de reemplazar "ruta_del_archivo.csv" con la ruta real del archivo

# Exploración rápida de datos
print("Train Summary:")
print(datos.describe())

# Estilo de vivienda
total_genres = datos['BldgType'].str.split('|', expand=True).stack().reset_index(drop=True)
total_genres.value_counts().plot(kind='bar')
plt.title("Estilo de vivienda")
plt.show()

# Año de construcción
total_genres = datos['YearBuilt'].astype(str).str.split('|', expand=True).stack().reset_index(drop=True)
total_genres.value_counts().plot(kind='bar')
plt.title("Año de construcción")
plt.show()

# Año de remodelación
total_genres = datos['YearRemodAdd'].astype(str).str.split('|', expand=True).stack().reset_index(drop=True)
total_genres.value_counts().plot(kind='bar')
plt.title("Año de remodelación")
plt.show()

# Cantidad de habitaciones
total_genres = datos['TotRmsAbvGrd'].astype(str).str.split('|', expand=True).stack().reset_index(drop=True)
total_genres.value_counts().plot(kind='bar')
plt.title("Cantidad de habitaciones")
plt.show()

# Cantidad de baños
total_genres = datos['FullBath'].astype(str).str.split('|', expand=True).stack().reset_index(drop=True)
total_genres.value_counts().plot(kind='bar')
plt.title("Cantidad de baños")
plt.show()

# Cantidad de cocinas
total_genres = datos['KitchenAbvGr'].astype(str).str.split('|', expand=True).stack().reset_index(drop=True)
total_genres.value_counts().plot(kind='bar')
plt.title("Cantidad de cocinas")
plt.show()

# Capacidad de garajes
total_genres = datos['GarageCars'].astype(str).str.split('|', expand=True).stack().reset_index(drop=True)
total_genres.value_counts().plot(kind='bar')
plt.title("Capacidad de garajes")
plt.show()