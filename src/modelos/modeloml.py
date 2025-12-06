import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.model_selection import cross_val_score
import numpy as np

class ModeloML:

    def regresion(self):
        df_data = pd.read_csv("")
        df_data.head()
        df_data.info()

        # Eliminamos la variable de salida(cantidad_pasajeros) y creamos un df de valores
        df_X = df_data.drop('cantidad_pasajeros', axis=1)
        df_X

        df_X = pd.get_dummies(df_X)
        df_X.head()

        X = df_X.values
        y = df_data['cantidad_pasajeros'].values


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

        reg = LinearRegression()

        reg.fit(X_train, y_train)

        cv_scores = cross_val_score(reg, X_train, y_train, cv=5)

        # Print the 5-fold cross-validation scores
        print(cv_scores)

        print("Average 5-Fold CV Score: {}".format(np.mean(cv_scores)))

        y_pred = reg.predict(X_test)
        y_pred

    def clasificacion(self):
        df_data = pd.read_csv("")
        df_data.head()
        df_data.info()








