import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, classification_report, \
    confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.tree import DecisionTreeClassifier, plot_tree
from src.datos.gestor_datos import GestorDatos


class ModeloML:
    def __init__(self):
        self.modelo = None
        self.modelo_arbol = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.columnas_features = None
        self.cuartiles = None
        self.capacidad_vehiculo = 35

    def regresion(self):
        """
        regresión lineal para predecir cantidad de pasajeros
        """
        # Cargar datos
        print("=" * 60)
        print("CARGANDO DATOS")
        print("=" * 60)
        cargador = GestorDatos(ruta_base="data/processed")
        df_data = cargador.cargar_csv("cartago.csv")

        print(f"\nDimensiones del dataset: {df_data.shape}")
        print(f"\nPrimeras filas:")
        print(df_data.head())

        print(f"\nInformación del dataset:")
        df_data.info()

        # Verificar valores nulos
        print(f"\nValores nulos por columna:")
        print(df_data.isnull().sum())

        # Preparar datos
        print("\n" + "=" * 60)
        print("PREPARANDO DATOS")
        print("=" * 60)

        df_X = df_data.drop('pasajerostotales', axis=1)

        #  dummies
        #df_X = pd.get_dummies(df_X, drop_first=True)

        print(f"\nNúmero de features después de encoding: {df_X.shape[1]}")
        print(f"Features: {list(df_X.columns)}")

        #  columnas para futuras predicciones
        self.columnas_features = df_X.columns.tolist()

        X = df_X.values
        y = df_data['pasajerostotales'].values

        #  train y test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\nTamaño del conjunto de entrenamiento: {self.X_train.shape[0]}")
        print(f"Tamaño del conjunto de prueba: {self.X_test.shape[0]}")

        self.modelo = LinearRegression()
        self.modelo.fit(self.X_train, self.y_train)

        print("\n✓ Modelo entrenado exitosamente")

        # Validación cruzada
        print("\n" + "=" * 60)
        print("VALIDACIÓN CRUZADA (5-FOLD)")
        print("=" * 60)

        cv_scores = cross_val_score(self.modelo, self.X_train, self.y_train,
                                    cv=5, scoring='r2')

        print(f"\nR² Scores por fold: {cv_scores}")
        print(f"R² Promedio: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Predicciones
        print("\n" + "=" * 60)

        y_train_pred = self.modelo.predict(self.X_train)
        self.y_pred = self.modelo.predict(self.X_test)

        # metricas de evaluación
        print("\n" + "=" * 60)
        print("MÉTRICAS DE EVALUACIÓN")
        print("=" * 60)

        # metricas en conjunto de entrenamiento
        rmse_train = np.sqrt(mean_squared_error(self.y_train, y_train_pred))
        mae_train = mean_absolute_error(self.y_train, y_train_pred)
        r2_train = r2_score(self.y_train, y_train_pred)

        print("\n--- CONJUNTO DE ENTRENAMIENTO ---")
        print(f"RMSE: {rmse_train:.2f}")
        print(f"MAE:  {mae_train:.2f}")
        print(f"R²:   {r2_train:.4f}")

        # metricas en conjunto de prueba
        rmse_test = np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        mae_test = mean_absolute_error(self.y_test, self.y_pred)
        r2_test = r2_score(self.y_test, self.y_pred)
        mape_test = np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100

        print("\n--- CONJUNTO DE PRUEBA ---")
        print(f"RMSE: {rmse_test:.2f} ← MÉTRICA PRINCIPAL")
        print(f"MAE:  {mae_test:.2f}")
        print(f"R²:   {r2_test:.4f}")
        print(f"MAPE: {mape_test:.2f}%")

        #  coeficientes
        print("\n" + "=" * 60)
        print("FEATURES MÁS IMPORTANTES")
        print("=" * 60)

        coeficientes = pd.DataFrame({
            'Feature': self.columnas_features,
            'Coeficiente': self.modelo.coef_
        })
        coeficientes['Abs_Coef'] = np.abs(coeficientes['Coeficiente'])
        coeficientes = coeficientes.sort_values('Abs_Coef', ascending=False)

        print(f"\nTop 10 features más influyentes:")
        print(coeficientes.head(10).to_string(index=False))

        # graficas
        self._generar_visualizaciones()

        # Guardar modelo
        self._guardar_modelo()

        return {
            'modelo': self.modelo,
            'rmse_test': rmse_test,
            'mae_test': mae_test,
            'r2_test': r2_test,
            'mape_test': mape_test,
            'coeficientes': coeficientes
        }

    def _generar_visualizaciones(self):
        """
        Genera visualizaciones del desempeño del modelo
        """
        print("\n" + "=" * 60)
        print("GENERANDO VISUALIZACIONES")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # predicciones vs Valores Reales
        axes[0, 0].scatter(self.y_test, self.y_pred, alpha=0.5, edgecolors='k')
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()],
                        [self.y_test.min(), self.y_test.max()],
                        'r--', lw=2, label='Predicción perfecta')
        axes[0, 0].set_xlabel('Valores Reales', fontsize=12)
        axes[0, 0].set_ylabel('Predicciones', fontsize=12)
        axes[0, 0].set_title('Predicciones vs Valores Reales', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # distribución de Residuos
        residuos = self.y_test - self.y_pred
        axes[0, 1].hist(residuos, bins=50, edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_xlabel('Residuos', fontsize=12)
        axes[0, 1].set_ylabel('Frecuencia', fontsize=12)
        axes[0, 1].set_title('Distribución de Residuos', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)

        # residuos vs Predicciones
        axes[1, 0].scatter(self.y_pred, residuos, alpha=0.5, edgecolors='k')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Predicciones', fontsize=12)
        axes[1, 0].set_ylabel('Residuos', fontsize=12)
        axes[1, 0].set_title('Residuos vs Predicciones', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)

        # Q-Q Plot para normalidad de residuos
        from scipy import stats
        stats.probplot(residuos, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normalidad de Residuos)', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('resultados_modelo_regresion.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráficos guardados en 'resultados_modelo_regresion.png'")
        plt.show()

    def _guardar_modelo(self):
        """
        Guarda el modelo entrenado
        """
        print("\n" + "=" * 60)
        print("GUARDANDO MODELO")
        print("=" * 60)

        # Guardar el modelo
        joblib.dump(self.modelo, 'modelo_regresion_lineal.pkl')
        print("✓ Modelo guardado en 'modelo_regresion_lineal.pkl'")

        # Guardar nombres de columnas
        joblib.dump(self.columnas_features, 'columnas_features.pkl')
        print("✓ Columnas guardadas en 'columnas_features.pkl'")

    def predecir(self, nuevos_datos):
        """
        Realiza predicciones con nuevos datos

        Args:
            nuevos_datos: DataFrame con las mismas columnas que los datos de entrenamiento

        Returns:
            Array con las predicciones
        """
        if self.modelo is None:
            raise ValueError("Primero debes entrenar el modelo con regresion()")

        # preparar datos
        df_X = pd.get_dummies(nuevos_datos, drop_first=True)

        # asegurar que tenga las mismas columnas que en entrenamiento
        for col in self.columnas_features:
            if col not in df_X.columns:
                df_X[col] = 0

        df_X = df_X[self.columnas_features]

        predicciones = self.modelo.predict(df_X.values)

        return predicciones

    def clasificacion(self):
        """
        Entrena modelos de clasificación para predecir nivel de ocupación
        """
        # Cargar datos
        print("=" * 60)
        print("CARGANDO DATOS")
        print("=" * 60)
        cargador = GestorDatos(ruta_base="data/processed")
        df_data = cargador.cargar_csv("cartago.csv")

        print(f"\nDimensiones del dataset: {df_data.shape}")
        print(f"\nPrimeras filas:")
        print(df_data.head())

        print(f"\nInformación del dataset:")
        df_data.info()

        # Calcular cuartiles
        self.cuartiles = df_data['pasajerostotales'].quantile([0.25, 0.5, 0.75]).values

        print(f"\nCuartiles calculados:")
        print(f"Q1 (25%): {self.cuartiles[0]:.0f} pasajeros")
        print(f"Q2 (50%): {self.cuartiles[1]:.0f} pasajeros")
        print(f"Q3 (75%): {self.cuartiles[2]:.0f} pasajeros")

        # Crear columna de ocupación basada en cuartiles
        print("\n" + "=" * 60)
        print("CREANDO VARIABLE DE OCUPACIÓN")
        print("=" * 60)

        def clasificar_ocupacion(pasajeros):
            if pasajeros <= self.cuartiles[0]:
                return 'Baja'
            elif pasajeros <= self.cuartiles[1]:
                return 'Media'
            elif pasajeros <= self.cuartiles[2]:
                return 'Alta'
            else:
                return 'Saturada'

        df_data['ocupacion'] = df_data['pasajerostotales'].apply(clasificar_ocupacion)

        #  distribución de ocupación
        print(f"\nDistribución de niveles de ocupación:")
        distribucion = df_data['ocupacion'].value_counts().sort_index()
        print(distribucion)
        print(f"\nPorcentajes:")
        print((distribucion / len(df_data) * 100).round(2))

        #  porcentaje de ocupación respecto a capacidad
        df_data['porcentaje_ocupacion'] = (df_data['pasajerostotales'] / self.capacidad_vehiculo * 100).round(2)

        print(f"\n--- Rangos de ocupación definidos ---")
        print(
            f"Baja:     0 - {self.cuartiles[0]:.0f} pasajeros (0% - {self.cuartiles[0] / self.capacidad_vehiculo * 100:.1f}% de capacidad)")
        print(
            f"Media:    {self.cuartiles[0]:.0f} - {self.cuartiles[1]:.0f} pasajeros ({self.cuartiles[0] / self.capacidad_vehiculo * 100:.1f}% - {self.cuartiles[1] / self.capacidad_vehiculo * 100:.1f}% de capacidad)")
        print(
            f"Alta:     {self.cuartiles[1]:.0f} - {self.cuartiles[2]:.0f} pasajeros ({self.cuartiles[1] / self.capacidad_vehiculo * 100:.1f}% - {self.cuartiles[2] / self.capacidad_vehiculo * 100:.1f}% de capacidad)")
        print(
            f"Saturada: >{self.cuartiles[2]:.0f} pasajeros (>{self.cuartiles[2] / self.capacidad_vehiculo * 100:.1f}% de capacidad)")

        #  datos para el modelo
        print("\n" + "=" * 60)
        print("PREPARANDO DATOS PARA EL MODELO")
        print("=" * 60)

        # a excluir
        columnas_excluir = ['ocupacion', 'pasajerostotales', 'porcentaje_ocupacion']

        # separar features y target
        df_X = df_data.drop(columnas_excluir, axis=1)

        # convertir variables categóricas a dummies
        df_X = pd.get_dummies(df_X, drop_first=True)

        print(f"\nNúmero de features después de encoding: {df_X.shape[1]}")
        print(f"Features seleccionadas: {list(df_X.columns)}")

        # guardar nombres de columnas
        self.columnas_features = df_X.columns.tolist()

        # Convertir a arrays
        X = df_X.values
        y = df_data['ocupacion'].values

        # Dividir en train y test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nTamaño del conjunto de entrenamiento: {self.X_train.shape[0]}")
        print(f"Tamaño del conjunto de prueba: {self.X_test.shape[0]}")

        print(f"\nDistribución en entrenamiento:")
        unique, counts = np.unique(self.y_train, return_counts=True)
        for clase, count in zip(unique, counts):
            print(f"  {clase}: {count} ({count / len(self.y_train) * 100:.1f}%)")

        # entrenar modelos
        resultados_arbol = self._entrenar_arbol_decision()
        resultados_rf = self._entrenar_random_forest()

        # comparar modelos
        self._comparar_modelos(resultados_arbol, resultados_rf)

        #  visualizaciones
        self._generar_visualizaciones()

        # Guardar modelos
        self._guardar_modelos()

        return {
            'arbol': resultados_arbol,
            'random_forest': resultados_rf,
            'cuartiles': self.cuartiles
        }

    def _entrenar_arbol_decision(self):
        """
        Arbol de Decisión
        """
        print("\n" + "=" * 60)
        print("ENTRENANDO ÁRBOL DE DECISIÓN")
        print("=" * 60)

        # Búsqueda de hiperparámetros
        param_grid = {
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }

        arbol_base = DecisionTreeClassifier(random_state=42)

        print("\nRealizando búsqueda de hiperparámetros...")
        grid_search = GridSearchCV(arbol_base, param_grid, cv=5,
                                   scoring='accuracy', n_jobs=-1, verbose=0)
        grid_search.fit(self.X_train, self.y_train)

        print(f"\nMejores hiperparámetros encontrados:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        # se usa mejor modelo
        self.modelo_arbol = grid_search.best_estimator_

        # validación cruzada
        cv_scores = cross_val_score(self.modelo_arbol, self.X_train, self.y_train,
                                    cv=5, scoring='accuracy')

        print(f"\n--- Validación Cruzada (5-fold) ---")
        print(f"Accuracy por fold: {cv_scores}")
        print(f"Accuracy promedio: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # predicciones
        y_train_pred = self.modelo_arbol.predict(self.X_train)
        y_test_pred = self.modelo_arbol.predict(self.X_test)


        print("\n--- MÉTRICAS EN CONJUNTO DE ENTRENAMIENTO ---")
        print(f"Accuracy: {accuracy_score(self.y_train, y_train_pred):.4f}")

        print("\n--- MÉTRICAS EN CONJUNTO DE PRUEBA ---")
        acc_test = accuracy_score(self.y_test, y_test_pred)
        print(f"Accuracy: {acc_test:.4f}")
        print(f"\nReporte de clasificación:")
        print(classification_report(self.y_test, y_test_pred))

        # importancia de features
        importancias = pd.DataFrame({
            'Feature': self.columnas_features,
            'Importancia': self.modelo_arbol.feature_importances_
        }).sort_values('Importancia', ascending=False)

        print(f"\nTop 10 Features más importantes:")
        print(importancias.head(10).to_string(index=False))

        return {
            'modelo': self.modelo_arbol,
            'accuracy_test': acc_test,
            'y_pred': y_test_pred,
            'importancias': importancias,
            'mejores_params': grid_search.best_params_
        }

    def _entrenar_random_forest(self):
        """
        entrena un modelo de Random Forest
        """
        print("\n" + "=" * 60)
        print("ENTRENANDO RANDOM FOREST")
        print("=" * 60)

        # bsqueda de hiperparámetros (más rápida para RF)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }

        rf_base = RandomForestClassifier(random_state=42)

        print("\nRealizando búsqueda de hiperparámetros...")
        grid_search = GridSearchCV(rf_base, param_grid, cv=5,
                                   scoring='accuracy', n_jobs=-1, verbose=0)
        grid_search.fit(self.X_train, self.y_train)

        print(f"\nMejores hiperparámetros encontrados:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        #  mejor modelo
        self.modelo_rf = grid_search.best_estimator_

        # aalidación cruzada
        cv_scores = cross_val_score(self.modelo_rf, self.X_train, self.y_train,
                                    cv=5, scoring='accuracy')

        print(f"\n--- Validación Cruzada (5-fold) ---")
        print(f"Accuracy por fold: {cv_scores}")
        print(f"Accuracy promedio: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

        # Predicciones
        y_train_pred = self.modelo_rf.predict(self.X_train)
        y_test_pred = self.modelo_rf.predict(self.X_test)

        # Métricas
        print("\n--- MÉTRICAS EN CONJUNTO DE ENTRENAMIENTO ---")
        print(f"Accuracy: {accuracy_score(self.y_train, y_train_pred):.4f}")

        print("\n--- MÉTRICAS EN CONJUNTO DE PRUEBA ---")
        acc_test = accuracy_score(self.y_test, y_test_pred)
        print(f"Accuracy: {acc_test:.4f}")
        print(f"\nReporte de clasificación:")
        print(classification_report(self.y_test, y_test_pred))

        # importancia de features
        importancias = pd.DataFrame({
            'Feature': self.columnas_features,
            'Importancia': self.modelo_rf.feature_importances_
        }).sort_values('Importancia', ascending=False)

        print(f"\nTop 10 Features más importantes:")
        print(importancias.head(10).to_string(index=False))

        return {
            'modelo': self.modelo_rf,
            'accuracy_test': acc_test,
            'y_pred': y_test_pred,
            'importancias': importancias,
            'mejores_params': grid_search.best_params_
        }

    def _comparar_modelos(self, resultados_arbol, resultados_rf):
        """
        desempeño de ambos modelos
        """
        print("\n" + "=" * 60)
        print("COMPARACIÓN DE MODELOS")
        print("=" * 60)

        comparacion = pd.DataFrame({
            'Modelo': ['Árbol de Decisión', 'Random Forest'],
            'Accuracy': [resultados_arbol['accuracy_test'],
                         resultados_rf['accuracy_test']]
        })

        print("\n", comparacion.to_string(index=False))

        mejor_modelo = 'Random Forest' if resultados_rf['accuracy_test'] > resultados_arbol[
            'accuracy_test'] else 'Árbol de Decisión'
        print(f"\n🏆 Mejor modelo: {mejor_modelo}")

    def _generar_visualizaciones(self):
        """
         visualizaciones de los resultados
        """
        print("\n" + "=" * 60)
        print("GENERANDO VISUALIZACIONES")
        print("=" * 60)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        #  matriz de confusión - Árbol
        y_pred_arbol = self.modelo_arbol.predict(self.X_test)
        cm_arbol = confusion_matrix(self.y_test, y_pred_arbol,
                                    labels=['Baja', 'Media', 'Alta', 'Saturada'])

        sns.heatmap(cm_arbol, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Baja', 'Media', 'Alta', 'Saturada'],
                    yticklabels=['Baja', 'Media', 'Alta', 'Saturada'],
                    ax=axes[0, 0])
        axes[0, 0].set_title('Matriz de Confusión - Árbol de Decisión',
                             fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Valor Real')
        axes[0, 0].set_xlabel('Predicción')

        # matriz de confusión - Random Forest
        y_pred_rf = self.modelo_rf.predict(self.X_test)
        cm_rf = confusion_matrix(self.y_test, y_pred_rf,
                                 labels=['Baja', 'Media', 'Alta', 'Saturada'])

        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
                    xticklabels=['Baja', 'Media', 'Alta', 'Saturada'],
                    yticklabels=['Baja', 'Media', 'Alta', 'Saturada'],
                    ax=axes[0, 1])
        axes[0, 1].set_title('Matriz de Confusión - Random Forest',
                             fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Valor Real')
        axes[0, 1].set_xlabel('Predicción')

        # importancia de features - Árbol
        importancias_arbol = pd.DataFrame({
            'Feature': self.columnas_features,
            'Importancia': self.modelo_arbol.feature_importances_
        }).sort_values('Importancia', ascending=False).head(10)

        axes[1, 0].barh(range(len(importancias_arbol)),
                        importancias_arbol['Importancia'], color='skyblue')
        axes[1, 0].set_yticks(range(len(importancias_arbol)))
        axes[1, 0].set_yticklabels(importancias_arbol['Feature'])
        axes[1, 0].set_xlabel('Importancia')
        axes[1, 0].set_title('Top 10 Features - Árbol de Decisión',
                             fontsize=14, fontweight='bold')
        axes[1, 0].invert_yaxis()

        # importancia de features - Random Forest
        importancias_rf = pd.DataFrame({
            'Feature': self.columnas_features,
            'Importancia': self.modelo_rf.feature_importances_
        }).sort_values('Importancia', ascending=False).head(10)

        axes[1, 1].barh(range(len(importancias_rf)),
                        importancias_rf['Importancia'], color='lightgreen')
        axes[1, 1].set_yticks(range(len(importancias_rf)))
        axes[1, 1].set_yticklabels(importancias_rf['Feature'])
        axes[1, 1].set_xlabel('Importancia')
        axes[1, 1].set_title('Top 10 Features - Random Forest',
                             fontsize=14, fontweight='bold')
        axes[1, 1].invert_yaxis()

        plt.tight_layout()
        plt.savefig('resultados_clasificacion.png', dpi=300, bbox_inches='tight')
        print("\n✓ Gráficos guardados en 'resultados_clasificacion.png'")
        plt.show()

        # visualizar árbol de decisión
        plt.figure(figsize=(25, 15))
        plot_tree(self.modelo_arbol,
                  feature_names=self.columnas_features,
                  class_names=['Alta', 'Baja', 'Media', 'Saturada'],
                  filled=True,
                  rounded=True,
                  fontsize=10,
                  max_depth=3)  # Limitamos profundidad para visualización
        plt.title('Árbol de Decisión (primeros 3 niveles)', fontsize=16, fontweight='bold')
        plt.savefig('arbol_decision.png', dpi=300, bbox_inches='tight')
        print("✓ Árbol de decisión guardado en 'arbol_decision.png'")
        plt.show()

    def _guardar_modelos(self):
        """
        guarda los modelos entrenados
        """
        print("\n" + "=" * 60)
        print("GUARDANDO MODELOS")
        print("=" * 60)

        joblib.dump(self.modelo_arbol, 'modelo_arbol_decision.pkl')
        print("✓ Árbol de decisión guardado en 'modelo_arbol_decision.pkl'")

        joblib.dump(self.modelo_rf, 'modelo_random_forest.pkl')
        print("✓ Random Forest guardado en 'modelo_random_forest.pkl'")

        joblib.dump(self.columnas_features, 'columnas_features_clasificacion.pkl')
        print("✓ Columnas guardadas en 'columnas_features_clasificacion.pkl'")

        joblib.dump(self.cuartiles, 'cuartiles_ocupacion.pkl')
        print("✓ Cuartiles guardados en 'cuartiles_ocupacion.pkl'")

    def predecir(self, nuevos_datos, usar_random_forest=True):
        """
        realiza predicciones con nuevos datos

        """
        modelo = self.modelo_rf if usar_random_forest else self.modelo_arbol

        if modelo is None:
            raise ValueError("Primero debes entrenar el modelo con clasificacion()")

        # preparar datos
        df_X = pd.get_dummies(nuevos_datos, drop_first=True)

        # asegurar que tenga las mismas columnas
        for col in self.columnas_features:
            if col not in df_X.columns:
                df_X[col] = 0

        df_X = df_X[self.columnas_features]

        # predecir
        predicciones = modelo.predict(df_X.values)
        probabilidades = modelo.predict_proba(df_X.values)

        return predicciones, probabilidades





