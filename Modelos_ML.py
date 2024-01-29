from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

def result_show():
    # Imprimir los resultados de cada partición y el promedio
    print(f'Puntajes de precisión de cada partición: {scores}')
    print(f'Precisión promedio: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})')

    # Imprimir los resultados de RandomForest
    print('\nResultados de RandomForest:')
    print(f'Puntajes de precisión de cada partición: {rf_scores}')
    print(f'Precisión promedio: {rf_scores.mean():.2f} (+/- {rf_scores.std() * 2:.2f})')


    # Imprimir los resultados de GBM (XGBoost)
    print('\nResultados de GBM (XGBoost):')
    print(f'Puntajes de precisión de cada partición: {gbm_scores}')
    print(f'Precisión promedio: {gbm_scores.mean():.2f} (+/- {gbm_scores.std() * 2:.2f})')



def graph():
    # Datos a graficar
    model_names = ['SVM', 'RandomForest', 'XGBoost']
    mean_scores = [scores.mean(), rf_scores.mean(), gbm_scores.mean()]
    std_scores = [scores.std() * 2, rf_scores.std() * 2, gbm_scores.std() * 2]

    # Crear gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(model_names, mean_scores, yerr=std_scores, capsize=10, color=['blue', 'green', 'orange'])
    plt.xlabel('Modelos')
    plt.ylabel('Precisión Promedio')
    plt.title('Precisión Promedio de Modelos')
    plt.ylim(0, 1)  # Establece el rango del eje y de 0 a 1 (precisión)

    # Mostrar gráfico
    plt.tight_layout()
    plt.show()


# Cargar los datos desde un archivo CSV
df = pd.read_csv('datos_voz.csv')


# Convertir etiquetas categóricas a numéricas
label_encoder_Labels = LabelEncoder()
label_encoder_Intensity = LabelEncoder()
label_encoder_Frecuency = LabelEncoder()


df['Labels'] = label_encoder_Labels.fit_transform(df['Labels'])
df['Intensity'] = label_encoder_Intensity.fit_transform(df['Intensity'])
df['Frecuency'] = label_encoder_Frecuency.fit_transform(df['Frecuency'])

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values




# Usamos StratifiedKFold para mantener el balance de clases en cada partición
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

svm_model = SVC(kernel='rbf', C=10, gamma=0.1)
scores = cross_val_score(svm_model, X, y, cv=cv, scoring='accuracy')

"""param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10]
}

grid_search = GridSearchCV(svm_model, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X, y)

 Después de ajustar, puedes obtener los mejores parámetros y el mejor estimador:
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_
print(best_params)
print(best_estimator)"""



# Entrenar y evaluar el modelo RandomForest
rf_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=20)
rf_scores = cross_val_score(rf_model, X, y, cv=cv, scoring='accuracy')

"""# Definir el espacio de parámetros para la búsqueda aleatoria
param_distributions = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Crear el objeto RandomizedSearchCV
random_search = RandomizedSearchCV(
    rf_model, param_distributions, 
    n_iter=100, cv=5, scoring='accuracy', 
    verbose=1, random_state=42, n_jobs=-1
)

# Realizar la búsqueda aleatoria (esto puede tardar un tiempo dependiendo del tamaño del conjunto de datos)
random_search.fit(X, y)

#Después de ajustar, puedes obtener los mejores parámetros y el mejor estimador:
best_params = random_search.best_params_
best_estimator = random_search.best_estimator_
print(best_params)
print(best_estimator)"""





# Entrenar y evaluar el modelo GBM (XGBoost)
gbm_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

"""# Definir el espacio de parámetros para la búsqueda aleatoria
param_distributions = {
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [3, 4, 5, 6, 7],
    'min_child_weight': [1, 2, 3, 4],
    'gamma': [0, 0.1, 0.2, 0.3, 0.4]
}

# Crear el objeto RandomizedSearchCV
random_search = RandomizedSearchCV(
    gbm_model, param_distributions, 
    n_iter=100, cv=5, scoring='accuracy', 
    verbose=1, random_state=42, n_jobs=-1
)


# Realizar la búsqueda aleatoria (esto puede tardar un tiempo dependiendo del tamaño del conjunto de datos)
random_search.fit(X, y)

# Después de ajustar, puedes obtener los mejores parámetros y el mejor estimador:
best_params = random_search.best_params_
best_estimator = random_search.best_estimator_

print(best_params)
print(best_estimator)"""

gbm_scores = cross_val_score(gbm_model, X, y, cv=cv, scoring='accuracy')



result_show()
graph()



