import pandas as pd
# Importar algoritmos de la librería sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class ModeloML:
    df_data = pd.read_csv("")
    df_data.head()
    df_data.info()

    # Eliminamos la variable de salida(clase) y creamos un df de valores
    X = df_data.drop('pasajeros', axis=1).values

    # Creamos la variable "clase" en un array de valores
    y = df_data['pasajeros'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23,
                                                        stratify=y)
    # stratify --> datos_etiquetados

    models = [('LR', LogisticRegression()),  # Creamos un array de todos los modelos a probar
              ('LDA', LinearDiscriminantAnalysis()),
              ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()),
              ('NB', GaussianNB()),
              ('SVM', SVC())]

    results = []
    names = []
    for name, model in models:
        # Divide los datos en 10 partes(9 folds de entrenamiento y 1 para probar )
        # Baraja los datos antes de dividir
        # El barajado es siempre igual (semilla 23)
        kf = KFold(n_splits=10, shuffle=True, random_state=23)

        # CV = crossvalidation
        cv_results = model_selection.cross_val_score(model, X, y, cv=kf)  #
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)


        # tomamos los mejores 3 modelos para comparar con cual llegamos al mejor resultado:









