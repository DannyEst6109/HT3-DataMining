import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# Cargar los datos
datos = pd.read_csv("train.csv")
test = pd.read_csv("test.csv", dtype=str)



# Seleccionar columnas
house = datos[['LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF',
               'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'TotRmsAbvGrd', 'Fireplaces',
               'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 'ScreenPorch',
               'PoolArea', 'MoSold', 'YrSold', 'SalePrice']]


# Eliminar filas con valores nulos
house = house.dropna()

# Resumen de los datos
print(house.describe())
#3. Análisis de grupo
# Análisis de grupos con k-medias
X = house.drop('SalePrice', axis=1)
kmeans = KMeans(n_clusters=3, random_state=123)
house['grupo'] = kmeans.fit_predict(X)

# Gráfico de los grupos
sns.scatterplot(x='GrLivArea', y='SalePrice', hue='grupo', data=house)
plt.show()

# Método de la silueta
silhouette_avg = silhouette_score(X, house['grupo'])
print(f"Silueta promedio: {silhouette_avg}")

# Análisis de grupos por cada cluster
for i in range(3):
    group_i = house[house['grupo'] == i]
    print(f"Proporción de casas en el grupo {i+1}:\n{group_i['SalePrice'].value_counts(normalize=True) * 100}")

# Resumen por grupo
for i in range(3):
    group_i = house[house['grupo'] == i]
    print(f"Resumen del grupo {i+1}:\n{group_i.describe()}")

# Similitud en las variables independientes y los precios de venta
correlations = house[['YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'TotRmsAbvGrd', 'GarageCars',
                      'GarageArea', 'SalePrice']].corr(method='pearson')
print(correlations)


#4. Divida el set de datos preprocesados en dos conjuntos: Entrenamiento y prueba. Describa el criterio que usó para crear los conjuntos: número de filas de cada uno, estratificado o no, balanceado o no, etc. Si le proveen un conjunto de datos de prueba y tiene suficientes datos, tómelo como de validación, pero haga sus propios conjuntos de prueba.

# División del conjunto de datos
train, test = train_test_split(datos, test_size=0.3, random_state=123)

# Eliminar columnas no deseadas
drop_cols = ["LotFrontage", "Alley", "MasVnrType", "MasVnrArea", "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1",
             "BsmtFinType2", "Electrical", "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish", "GarageQual",
             "GarageCond", "PoolQC", "Fence", "MiscFeature"]
train = train.drop(drop_cols, axis=1)
test = test.drop(drop_cols, axis=1)


### 5. Haga ingeniería de características, ¿qué variables cree que puedan ser mejores predictores para el precio de las casas? Explique en que basó la selección o no de las variables.

# Ingeniería de características
selected_features = ['LotArea', 'Neighborhood', 'BldgType', 'OverallQual', 'FullBath', 'TotRmsAbvGrd', 'Fireplaces',
                      'GarageCars']
X_train = train[selected_features]
y_train = train['SalePrice']
X_test = test[selected_features]
y_test = test['SalePrice']


### 7. Seleccione una de las variables y haga un modelo univariado de regresión lineal para predecir el precio de las casas. Analice el modelo 

# Modelo univariado de regresión lineal
model_univariate = LinearRegression()
X_train_univariate = X_train[['LotArea']]
X_test_univariate = X_test[['LotArea']]
model_univariate.fit(X_train_univariate, y_train)
y_pred_univariate = model_univariate.predict(X_test_univariate)

# Análisis del modelo univariado
print(model_univariate.coef_)
print(model_univariate.intercept_)
print("R2:", r2_score(y_test, y_pred_univariate))

# Gráfico del modelo univariado
plt.scatter(X_test['LotArea'], y_test, color='gray')
plt.plot(X_test['LotArea'], y_pred_univariate, color='red', linewidth=2)
plt.xlabel('LotArea')
plt.ylabel('SalePrice')
plt.title('Modelo Univariado de Regresión Lineal')
plt.show()


### 8. Haga un modelo de regresión lineal con todas las variables numéricas para predecir el precio de las casas. Analice el modelo (resumen, residuos, resultados de la predicción). Muestre el modelo gráficamente.
# Suponiendo que 'house' es tu DataFrame y que 'SalePrice' es tu variable objetivo

# Extraer las variables independientes (X) y la variable dependiente (y)
X = house.drop('SalePrice', axis=1)
y = house['SalePrice']

# Convertir variables categóricas en variables dummy
X = pd.get_dummies(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=123)

# Crear y entrenar el modelo de regresión lineal
model_multivariate = LinearRegression()
model_multivariate.fit(X_train, y_train)

# Modelo de regresión lineal con todas las variables numéricas
X_train_multivariate = train.drop(['SalePrice'], axis=1)
X_test_multivariate = test.drop(['SalePrice'], axis=1)
model_multivariate = LinearRegression()
model_multivariate.fit(X_train_multivariate, y_train)
y_pred_multivariate = model_multivariate.predict(X_test_multivariate)

### 9. Analice el modelo. Determine si hay multicolinealidad entre las variables, y cuáles son las que aportan al modelo, por su valor de significación. Haga un análisis de correlación de las características del modelo y especifique si el modelo se adapta bien a los datos. Explique si hay sobreajuste (overfitting) o no. En caso de existir sobreajuste, haga otro modelo que lo corrija.

# Análisis del modelo multivariado
print("Coeficientes:", model_multivariate.coef_)
print("Intercepto:", model_multivariate.intercept_)
print("R2:", r2_score(y_test, y_pred_multivariate))

### 10. Si tiene multicolinealidad o sobreajuste, haga un modelo con las variables que sean mejores predictoras del precio de las casas. Determine la calidad del modelo realizando un análisis de los residuos. Muéstrelo gráficamente.

# Gráfico del modelo multivariado
plt.scatter(y_test, y_pred_multivariate, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], linestyle='--', color='red', linewidth=2)
plt.xlabel('Precio Real')
plt.ylabel('Precio Predicho')
plt.title('Modelo de Regresión Lineal Multivariado')
plt.show()

# Análisis de multicolinealidad
vif_data = pd.DataFrame()
vif_data["Variable"] = X_train_multivariate.columns
vif_data["VIF"] = [variance_inflation_factor(X_train_multivariate.values, i) for i in range(X_train_multivariate.shape[1])]
print(vif_data)

### 11. Utilice cada modelo con el conjunto de prueba y determine la eficiencia del algoritmo para predecir el precio de las casas. ¿Qué tan bien lo hizo?

# División del conjunto de datos para el modelo corregido
X_train_corr, X_test_corr, y_train_corr, y_test_corr = train_test_split(datosCC.drop(['SalePrice', 'clasificacion', 'y'], axis=1), datosCC['SalePrice'], test_size=0.3, random_state=123)

# Modelo de regresión lineal corregido
model_corr = LinearRegression()
model_corr.fit(X_train_corr, y_train_corr)
y_pred_corr = model_corr.predict(X_test_corr)

# Análisis del modelo corregido
print("Coeficientes:", model_corr.coef_)
print("Intercepto:", model_corr.intercept_)
print("R2:", r2_score(y_test_corr, y_pred_corr))

# Gráfico del modelo corregido
plt.scatter(y_test_corr, y_pred_corr, color='green')
plt.plot([min(y_test_corr), max(y_test_corr)], [min(y_test_corr), max(y_test_corr)], linestyle='--', color='red', linewidth=2)
plt.xlabel('Precio Real')
plt.ylabel('Precio Predicho')
plt.title('Modelo de Regresión Lineal Corregido')
plt.show()