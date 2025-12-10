import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class VisualizadorDatos:

    def __init__(self, ruta_base=None):
        print("===== INICIALIZANDO VISUALIZADOR =====")

        if ruta_base is None:
            ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.ruta_base = ruta_base

        self.ruta_csv = os.path.join(self.ruta_base, "data", "processed", "cartago.csv")
        print(f"Ruta detectada: {self.ruta_csv}")

        if not os.path.isfile(self.ruta_csv):
            print("ERROR: no se encontró el CSV.")
            self.df = None
        else:
            self.df = pd.read_csv(self.ruta_csv)
            print("\nColumnas detectadas en el dataset:")
            print(self.df.columns.tolist())

    # -------------------------------------------------------------------------
    # GRÁFICO 1: Pasajeros totales por año
    # -------------------------------------------------------------------------
    def pasajeros_por_anio(self):
        if self.df is None:
            return

        df_group = self.df.groupby("year")["pasajerostotales"].sum()

        plt.figure(figsize=(10, 6))
        plt.plot(df_group.index, df_group.values, marker="o")
        plt.title("Pasajeros Totales por Año")
        plt.xlabel("Año")
        plt.ylabel("Total de Pasajeros")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # GRÁFICO 2: Pasajeros por mes (promedio)
    # -------------------------------------------------------------------------
    def pasajeros_por_mes(self):
        if self.df is None:
            return

        df_group = self.df.groupby("month")["pasajerostotales"].mean()

        plt.figure(figsize=(10, 6))
        plt.bar(df_group.index, df_group.values)
        plt.title("Promedio de Pasajeros por Mes")
        plt.xlabel("Mes")
        plt.ylabel("Promedio de Pasajeros")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # GRÁFICO 3: Top 10 rutas con más pasajeros
    # -------------------------------------------------------------------------
    def top_rutas(self):
        if self.df is None:
            return

        df_group = (self.df.groupby("nombreruta")["pasajerostotales"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10))

        plt.figure(figsize=(12, 6))
        df_group.plot(kind="bar", color="purple")
        plt.title("Top 10 Rutas con Más Pasajeros")
        plt.xlabel("Ruta")
        plt.ylabel("Total Pasajeros")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # GRÁFICO 4: Adultos mayores vs regulares (totales)
    # -------------------------------------------------------------------------
    def adultos_vs_regulares(self):
        if self.df is None:
            return

        total_adultos = self.df["pasajerosadultomayor"].sum()
        total_regulares = self.df["pasajerosregulares"].sum()

        plt.figure(figsize=(8, 6))
        plt.bar(["Adulto Mayor", "Regulares"], [total_adultos, total_regulares])
        plt.title("Comparación de Pasajeros por Tipo")
        plt.ylabel("Total Pasajeros")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # GRÁFICO 5: Distribución de pasajeros totales
    # -------------------------------------------------------------------------
    def distribucion_pasajeros(self):
        if self.df is None:
            return

        plt.figure(figsize=(10, 6))
        sns.histplot(self.df["pasajerostotales"], kde=True)
        plt.title("Distribución de Pasajeros Totales")
        plt.xlabel("Pasajeros Totales")
        plt.tight_layout()
        plt.show()

    # -------------------------------------------------------------------------
    # GRÁFICO 6: Matriz de correlación numérica
    # -------------------------------------------------------------------------
    def matriz_correlacion(self):
        if self.df is None:
            return

        columnas_numericas = [
            "year", "month", "pasajerosadultomayor",
            "pasajerosregulares", "pasajerostotales"
        ]

        plt.figure(figsize=(10, 7))
        sns.heatmap(self.df[columnas_numericas].corr(), annot=True, cmap="coolwarm")
        plt.title("Matriz de Correlación")
        plt.tight_layout()
        plt.show()


# =============================================================================
# EJECUCIÓN PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    viz = VisualizadorDatos()

    if viz.df is not None:
        print("\n===== GENERANDO GRÁFICOS =====\n")

        viz.pasajeros_por_anio()
        viz.pasajeros_por_mes()
        viz.top_rutas()
        viz.adultos_vs_regulares()
        viz.distribucion_pasajeros()
        viz.matriz_correlacion()
