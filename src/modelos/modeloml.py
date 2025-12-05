import pandas as pd
# Importar algoritmos de la librería sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class ModeloML:
    print("Hola")
    df_data = pd.read_csv("")
    df_data.head()
    df_data.info()

    # Eliminamos la variable de salida(clase) y creamos un df de valores
    X = df_data.drop('pasajeros', axis=1).values

    # Creamos la variable "clase" en un array de valores
    y = df_data['pasajeros'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,
                                                        stratify=y)
    # stratify --> datos_etiquetados
    







