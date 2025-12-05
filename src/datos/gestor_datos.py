import pandas as pd
from pathlib import Path
from typing import Optional

class GestorDatos:
    #cargar y dar info basica del csv
    def __init__(self, ruta_base: str = "data/raw"):
        self._ruta_base = Path(ruta_base)
        self._dataframe: Optional[pd.DataFrame] = None
        self._nombre_archivo: Optional[str] = None
        self._num_filas: int = 0
        self._num_columnas: int = 0
        self._porcentaje_nulos: float = 0.0

    #getters
    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        return self._dataframe

    @property
    def nombre_archivo(self) -> Optional[str]:
        return self._nombre_archivo

    @property
    def num_filas(self) -> int:
        return self._num_filas

    @property
    def num_columnas(self) -> int:
        return self._num_columnas

    @property
    def porcentaje_nulos(self) -> float:
        return self._porcentaje_nulos

    @property
    def ruta_base(self) -> Path:
        return self._ruta_base

    @ruta_base.setter
    def ruta_base(self, value: str):
        self._ruta_base = Path(value)


    def cargar_csv(
            self,
            nombre_archivo: str,
            encoding: str = 'utf-8',
            delimiter: str = ',',
            decimal='.'
    ) -> pd.DataFrame:

        ruta_completa = self._ruta_base/nombre_archivo

        #el archivo existe?
        #uso de f string para combinar texto con variables
        if not ruta_completa.exists():
            print(f"Archivo no encontrado: {ruta_completa}")
            raise FileNotFoundError(f"No se encontró el archivo: {ruta_completa}")

        print(f"Iniciando carga de: {nombre_archivo}")

        try:
            #cargar el csv
            self._dataframe = pd.read_csv(
                ruta_completa,
                encoding=encoding,
                delimiter=delimiter,
                decimal=decimal
            )

            #esta vacío?
            if self._dataframe.empty:
                raise ValueError(f"El archivo {nombre_archivo} está vacío")

            self._nombre_archivo = nombre_archivo

            #métricas
            self.calcular_metricas()

            #información de carga
            self.registrar_info_carga()

            print(f"Carga exitosa de {nombre_archivo}")

            return self._dataframe

        except pd.errors.EmptyDataError:
            self._logger.error(f"El archivo {nombre_archivo} está vacío o mal formado")
            raise
        except Exception as e:
            self._logger.error(f"Error al cargar {nombre_archivo}: {str(e)}")
            raise

    def calcular_metricas(self):
        if self._dataframe is None:
            return

        #dimensiones
        self._num_filas = len(self._dataframe)
        self._num_columnas = len(self._dataframe.columns)

        # % total de nulos
        total_valores = self._dataframe.size
        total_nulos = self._dataframe.isnull().sum().sum() # sum() por columna, sum() todas las columnas
        self._porcentaje_nulos = (total_nulos / total_valores) * 100 if total_valores > 0 else 0.0 # evitar division entre 0


    def registrar_info_carga(self):
        print("=" * 60)
        print("PROYECTO PROGRAMACION 2: INFORMACION ARCHIVO CARGADO")
        print("=" * 60)
        print(f"Archivo: {self._nombre_archivo}")
        print(f"Número de filas: {self._num_filas:,}")
        print(f"Número de columnas: {self._num_columnas}")
        print(f"Porcentaje global de nulos: {self._porcentaje_nulos:.2f}%")
        print("-" * 60)
        print(f"Primeras 5 filas: \n{self.dataframe.head(5)}")
        print("-" * 60)
        print(f"Ultimas 5 filas: \n{self.dataframe.tail(5)}")

