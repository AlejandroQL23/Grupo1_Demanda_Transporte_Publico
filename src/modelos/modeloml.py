import pandas as pd
import numpy as np
from sklearn.ensemble import  RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
from xgboost import XGBRegressor
from src.datos.gestor_datos import GestorDatos


class ModeloML:

    def __init__(self):
        self.modelos_regresion = {}
        self.modelos_clasificacion = {}
        self.scaler_regresion = None
        self.scaler_clasificacion = None
        self.resultados_regresion = {}
        self.resultados_clasificacion = {}

    def regresion(self):
        """
         multiples algoritmos y compara
        """
        # 1 cargar y preparar datos
        cargador = GestorDatos(ruta_base="data/processed")
        df_data = cargador.transformar("cartago.csv")

        print(f"\nDatos cargados: {df_data.shape[0]} registros, {df_data.shape[1]} columnas")
        df_data.info()

        # 2 entrada y objetivo
        # variables de entrada usadas para mejorar resultados
        entradas = [
            'month', 'dia_semana', 'es_fin_semana',  # relacion dias
            'es_feriado_mes', 'cant_feriados', 'feriado_finde',   # dias especiales
            'precipitacion_mm', 'temperatura_c',  # clima
            'ruta_promedio_pasajeros', 'proporcion_adultos_mayor',  #  ruta
            'year_norm'
        ]

        X = df_data[entradas]
        y = df_data['pasajerostotales']  #  objetivo

        print(f"\nFeatures utilizadas: {len(entradas)}")
        print(f"Target: pasajerostotales (rango: {y.min():.0f} - {y.max():.0f})")

        # 3 train-test (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\nConjunto de entrenamiento: {X_train.shape[0]} registros")
        print(f"Conjunto de prueba: {X_test.shape[0]} registros")

        # 4 mormalización de features
        self.scaler_regresion = StandardScaler()
        X_train_scaled = self.scaler_regresion.fit_transform(X_train)
        X_test_scaled = self.scaler_regresion.transform(X_test)

        # 5  modelos
        modelos = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=42),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'Random Forest': RandomForestRegressor(
                n_estimators=70, # numero de arboles
                max_depth=7, # profundidad del arbol
                min_samples_split=3, # muestras min para dividir un nodo
                random_state=42,
                n_jobs=-1 # usa todos los nucleos
            ),
            'XGBoost': XGBRegressor(
                n_estimators=59, # numero de arboles
                max_depth=9, # profundidad
                learning_rate=0.1, # aprendizaje (a + bajo + estable + arboles)
                random_state=42,
                n_jobs=-1 # usa todos lo nucleos
            )
        }

        # 6 entrenar y evaluar

        for nombre, modelo in modelos.items():
            print(f"\n--- {nombre} ---")

            # entrenar
            if nombre in ['KNN', 'Linear Regression', 'Ridge Regression']:
                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)
            else:
                modelo.fit(X_train, y_train)
                y_pred = modelo.predict(X_test)

            # metricas
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # resultados
            self.modelos_regresion[nombre] = modelo
            self.resultados_regresion[nombre] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'y_pred': y_pred
            }

            print(f"  RMSE: {rmse:.2f} pasajeros")
            print(f"  MAE:  {mae:.2f} pasajeros")
            print(f"  R²:   {r2:.4f}")

        # 7 mejor modelo
        mejor_modelo = min(self.resultados_regresion.items(),
                           key=lambda x: x[1]['RMSE'])

        print("\n" + "=" * 60)
        print(f"MEJOR MODELO: {mejor_modelo[0]}")
        print(f"RMSE: {mejor_modelo[1]['RMSE']:.2f} pasajeros")
        print(f"R²: {mejor_modelo[1]['R2']:.4f}")
        print("=" * 60)

        # 8 resultados
        self._visualizar_regresion()

        # 9 importancia de features
        if 'Random Forest' in self.modelos_regresion:
            self._importancia_features_regresion(entradas)

        # 10. Guardar mejor modelo
        joblib.dump(self.modelos_regresion[mejor_modelo[0]],
                    'mejor_modelo_regresion.pkl')
        joblib.dump(self.scaler_regresion, 'scaler_regresion.pkl')
        print("\nModelo guardado en: mejor_modelo_regresion.pkl")

        return self.resultados_regresion


    def clasificacion(self):
        """
        modelo de clasificacion para predecir el nivel de ocupacion del servicio.
         Baja, Media, Alta, Saturada (basado en cuartiles)
        """

        # 1 datos
        cargador = GestorDatos(ruta_base="data/processed")
        df_data = cargador.transformar("cartago.csv")

        print(f"\nDatos cargados: {df_data.shape[0]} registros")

        # 2 variable objetivo usando cuartiles 25% cada uno, usando tot pax
        q25 = df_data['pasajerostotales'].quantile(0.25)  # Cuartil 1 (25%)
        q50 = df_data['pasajerostotales'].quantile(0.50)  # Cuartil 2 (50% - mediana)
        q75 = df_data['pasajerostotales'].quantile(0.75)  # Cuartil 3 (75%)

        print(f"\nCuartiles calculados:")
        print(f"  Q1 (25%): {q25:.0f} pasajeros")
        print(f"  Q2 (50%): {q50:.0f} pasajeros")
        print(f"  Q3 (75%): {q75:.0f} pasajeros")
        print(f"  Máximo:   {df_data['pasajerostotales'].max():.0f} pasajeros")

        # clasificar
        def clasificar_ocupacion_cuartiles(pasajeros):
            if pasajeros <= q25:
                return 0  # Baja (0-25%)
            elif pasajeros <= q50:
                return 1  # Media (25-50%)
            elif pasajeros <= q75:
                return 2  # Alta (50-75%)
            else:
                return 3  # Saturada (75-100%)

        df_data['nivel_ocupacion'] = df_data['pasajerostotales'].apply(
            clasificar_ocupacion_cuartiles
        )

        # Mapeo para visualización
        mapeo_ocupacion = {0: 'Baja', 1: 'Media', 2: 'Alta', 3: 'Saturada'}

        print("\nDistribución de niveles de ocupación:")
        print(df_data['nivel_ocupacion'].value_counts().sort_index())

        # 3 entradas y objetivo
        entradas = [
            'pasajerostotales', 'pasajerosregulares', 'pasajerosadultomayor',
            'month', 'dia_semana', 'es_fin_semana',
            'ruta_promedio_pasajeros',
            'es_feriado_mes', 'precipitacion_mm', 'temperatura_c',
        ]

        X = df_data[entradas]
        y = df_data['nivel_ocupacion']

        print(f"\nFeatures utilizadas: {len(entradas)}")
        print(f"Clases objetivo: {list(mapeo_ocupacion.values())}")

        # 4 train-test (80-20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"\nConjunto de entrenamiento: {X_train.shape[0]} registros")
        print(f"Conjunto de prueba: {X_test.shape[0]} registros")

        # 5 normalizacion
        self.scaler_clasificacion = StandardScaler()
        X_train_scaled = self.scaler_clasificacion.fit_transform(X_train)
        X_test_scaled = self.scaler_clasificacion.transform(X_test)

        # 6 modelo
        modelos = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, # numero max de iteraciones
                random_state=42
            )
        }

        # 7 entrenar y evaluar

        for nombre, modelo in modelos.items():
            print(f"\n--- {nombre} ---")

            # entrenar
            if nombre == 'Logistic Regression':
                modelo.fit(X_train_scaled, y_train)
                y_pred = modelo.predict(X_test_scaled)

            # metricas
            accuracy = accuracy_score(y_test, y_pred)

            clases_presentes = sorted(y_test.unique())
            nombres_clases_presentes = [mapeo_ocupacion[c] for c in clases_presentes]

            #  resultados
            self.modelos_clasificacion[nombre] = modelo
            self.resultados_clasificacion[nombre] = {
                'accuracy': accuracy,
                'y_pred': y_pred,
                'clases_presentes': clases_presentes,
                'classification_report': classification_report(
                    y_test, y_pred,
                    labels=clases_presentes,
                    target_names=nombres_clases_presentes,
                    output_dict=True,
                    zero_division=0
                ),
                'confusion_matrix': confusion_matrix(y_test, y_pred, labels=clases_presentes)
            }

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Clases detectadas: {nombres_clases_presentes}")
            print("\nReporte de clasificación:")
            print(classification_report(
                y_test, y_pred,
                labels=clases_presentes,
                target_names=nombres_clases_presentes,
                zero_division=0
            ))

        # 8 mejor modelo
        mejor_modelo = max(self.resultados_clasificacion.items(),
                           key=lambda x: x[1]['accuracy'])

        print("\n" + "=" * 60)
        print(f"MEJOR MODELO: {mejor_modelo[0]}")
        print(f"Accuracy: {mejor_modelo[1]['accuracy']:.4f}")
        print("=" * 60)

        # 9 Guardar
        joblib.dump(self.modelos_clasificacion[mejor_modelo[0]],
                    'mejor_modelo_clasificacion.pkl')
        joblib.dump(self.scaler_clasificacion, 'scaler_clasificacion.pkl')
        print("\nModelo guardado en: mejor_modelo_clasificacion.pkl")

        return self.resultados_clasificacion

    def _visualizar_regresion(self):
        """Visualiza resultados """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # comparación
        modelos_nombres = list(self.resultados_regresion.keys())
        rmse_valores = [self.resultados_regresion[m]['RMSE'] for m in modelos_nombres]
        r2_valores = [self.resultados_regresion[m]['R2'] for m in modelos_nombres]

        axes[0, 0].bar(modelos_nombres, rmse_valores, color='steelblue')
        axes[0, 0].set_title('RMSE por Modelo', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)

        axes[0, 1].bar(modelos_nombres, r2_valores, color='coral')
        axes[0, 1].set_title('R² por Modelo', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('R²')
        axes[0, 1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig('regresion_resultados.png', dpi=300, bbox_inches='tight')
        print("\nGráficas guardadas en: regresion_resultados.png")
        plt.close()


    def _importancia_features_regresion(self, features):
        """ importancia de entradas para regresión"""
        rf_model = self.modelos_regresion['Random Forest']
        importancias = rf_model.feature_importances_
        indices = np.argsort(importancias)[::-1][:10]  # Top 10

        plt.figure(figsize=(10, 6))
        plt.bar(range(10), importancias[indices], color='steelblue')
        plt.xticks(range(10), [features[i] for i in indices], rotation=45, ha='right')
        plt.title('Top 10 Features - Importancia (Regresión)',
                  fontsize=12, fontweight='bold')
        plt.ylabel('Importancia')
        plt.tight_layout()
        plt.savefig('feature_importance_regresion.png', dpi=300)
        print("Importancia de features guardada en: feature_importance_regresion.png")
        plt.close()

    def _importancia_features_clasificacion(self, features):
        """ importancia de entrdas para clasificación"""
        rf_model = self.modelos_clasificacion['Random Forest']
        importancias = rf_model.feature_importances_
        indices = np.argsort(importancias)[::-1][:10]

        plt.figure(figsize=(10, 6))
        plt.bar(range(10), importancias[indices], color='coral')
        plt.xticks(range(10), [features[i] for i in indices], rotation=45, ha='right')
        plt.title('Top 10 Features - Importancia (Clasificación)',
                  fontsize=12, fontweight='bold')
        plt.ylabel('Importancia')
        plt.tight_layout()
        plt.savefig('feature_importance_clasificacion.png', dpi=300)
        print("Importancia de features guardada en: feature_importance_clasificacion.png")
        plt.close()