import pandas as pd
import numpy as np
import os

class ProcesadorEDA:
    def __init__(self, ruta_archivo: str):
        self.ruta = ruta_archivo
        self.df = None

    def cargar_datos(self):
        print(f"\nIntentando cargar archivo desde:\n{self.ruta}\n")

        if not os.path.exists(self.ruta):
            print("ERROR: No se encontró el archivo.")
            return

        try:
            self.df = pd.read_csv(self.ruta)
            print("===== DATOS CARGADOS CORRECTAMENTE =====")
            print(self.df.head())
        except Exception as e:
            print(f"ERROR inesperado al cargar el archivo: {e}")

    def info_general(self):
        print("\n===== INFORMACIÓN GENERAL =====")
        print(self.df.info())

    def estadisticas_generales(self):
        print("\n===== ESTADÍSTICAS GENERALES =====")
        print(self.df.describe(include='all'))

    def revisar_nulos(self):
        print("\n===== VALORES NULOS =====")
        nulos = self.df.isnull().sum()
        porcentaje = (self.df.isnull().mean() * 100).round(2)
        print(pd.DataFrame({"Nulos": nulos, "%": porcentaje}))

    def tipos_datos(self):
        print("\n===== TIPOS DE DATOS =====")
        print(self.df.dtypes)

    def revisar_duplicados(self):
        print("\n===== DUPLICADOS =====")
        duplicados = self.df.duplicated().sum()
        print(f"Filas duplicadas: {duplicados}")

    def estadisticas_numericas(self):
        print("\n===== ESTADÍSTICAS NUMÉRICAS =====")
        print(self.df.select_dtypes(include=np.number).describe())

    def estadisticas_categoricas(self):
        print("\n===== COLUMNAS CATEGÓRICAS =====")
        categoricas = self.df.select_dtypes(include='object')
        for col in categoricas.columns:
            print(f"\n--- {col} ---")
            print(categoricas[col].value_counts())

    def resumen_completo(self):
        print("\n\n=========== INICIO DEL ANÁLISIS EDA ===========")
        self.cargar_datos()

        if self.df is None:
            print("No se pudo cargar el dataset. EDA cancelado.")
            return

        self.info_general()
        self.tipos_datos()
        self.estadisticas_generales()
        self.estadisticas_numericas()
        self.estadisticas_categoricas()
        self.revisar_nulos()
        self.revisar_duplicados()

        print("\n=========== FIN DEL EDA ===========")


def buscar_csv(base_path):
    print("\nBuscando archivos CSV en tu proyecto...\n")

    archivos_csv = []
    for root, dirs, files in os.walk(base_path):
        for f in files:
            if f.endswith(".csv"):
                archivos_csv.append(os.path.join(root, f))

    if not archivos_csv:
        print(" No se encontraron archivos CSV en el proyecto.")
        return None

    print("Archivos CSV encontrados:")
    for i, ruta in enumerate(archivos_csv):
        print(f"{i + 1}. {ruta}")

    print("\nSelecciona el número del archivo que quieres usar:")
    seleccion = int(input("Número: ")) - 1
    return archivos_csv[seleccion]


if __name__ == "__main__":
    BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Directorio base detectado:", BASE)

    ruta_csv = buscar_csv(BASE)

    if ruta_csv:
        eda = ProcesadorEDA(ruta_csv)
        eda.resumen_completo()


