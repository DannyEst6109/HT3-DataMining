import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar datos
datos = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv", dtype={'MSSubClass': 'object', 'MoSold': 'object', 'YrSold': 'object'})

# Selección de columnas
columns_to_keep = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                   'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'TotRmsAbvGrd',
                   'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
                   'EnclosedPorch', 'ScreenPorch', 'PoolArea', 'MoSold', 'YrSold', 'SalePrice']

house = datos[columns_to_keep].dropna()

# Resumen
print(house.describe())

# Análisis exploratorio

# División de conjuntos de entrenamiento y prueba
train, test = train_test_split(house, test_size=0.3, random_state=123)

# Modelo de regresión lineal univariado
modelo_univariado = LinearRegression().fit(train[['LotArea']], train['SalePrice'])
prediccion_univariada = modelo_univariado.predict(test[['LotArea']])

# Evaluación del modelo univariado
print(f'Coeficiente: {modelo_univariado.coef_[0]:.2f}')
print(f'R-squared: {r2_score(test["SalePrice"], prediccion_univariada):.2f}')

# Gráfico del modelo univariado
sns.scatterplot(x=test['LotArea'], y=test['SalePrice'])
plt.plot(test['LotArea'], prediccion_univariada, color='red')
plt.xlabel('LotArea')
plt.ylabel('SalePrice')
plt.title('Regresión Lineal Univariada')
plt.show()

# Modelo de regresión lineal multivariado
features = ['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
            'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'TotRmsAbvGrd',
            'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF']
modelo_multivariado = LinearRegression().fit(train[features], train['SalePrice'])
prediccion_multivariada = modelo_multivariado.predict(test[features])

# Evaluación del modelo multivariado
print(f'R-squared: {r2_score(test["SalePrice"], prediccion_multivariada):.2f}')

# Gráfico del modelo multivariado
sns.scatterplot(x=test['SalePrice'], y=prediccion_multivariada)
plt.xlabel('SalePrice Real')
plt.ylabel('SalePrice Predicho')
plt.title('Regresión Lineal Multivariada')
plt.show()