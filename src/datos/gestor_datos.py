# gestor_datos.py
"""
Gestor de datos mejorado para el Proyecto de Demanda de Transporte Público.

Provee:
- Carga robusta de CSVs (soporte chunks, parse_dates, usecols, etc.)
- Cálculo de métricas básicas (filas, columnas, % nulos)
- Resumen y validación de columnas esperadas
- Filtrado por provincia/cantón y selección de columnas
- Guardado de dataframes procesados

Uso básico:
    gestor = GestorDatos(ruta_base="src", verbose=True)
    df = gestor.cargar_csv("datos.csv", parse_dates=["fecha"])
    gestor.seleccionar_columnas(["year","month","nombre","codigoruta","nombreruta"], in_place=True)
    gestor.filtrar_por_provincia("provincia", "Cartago", in_place=True)
    gestor.guardar_dataframe_procesado("data/processed/datos_cartago.csv")
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import pandas as pd
import logging
import os

from src.api.cliente_api import ClienteAPI


class GestorDatos:
    def __init__(
        self,
        ruta_base: str = "data/raw",
        logger: Optional[logging.Logger] = None,
        verbose: bool = True
    ):
        """
        Inicializa el gestor.

        :param ruta_base: directorio base donde buscar archivos por defecto (puede ser relativo o absoluto)
        :param logger: logger opcional; si es None se crea uno básico
        :param verbose: si True muestra mensajes por consola
        """
        self._ruta_base: Path = Path(ruta_base)
        self._dataframe: Optional[pd.DataFrame] = None
        self._nombre_archivo: Optional[str] = None
        self._num_filas: int = 0
        self._num_columnas: int = 0
        self._porcentaje_nulos: float = 0.0
        self.verbose = verbose

        # Logger: si no se pasa, crear uno básico para esta clase
        if logger is None:
            self._logger = logging.getLogger("GestorDatos")
            if not self._logger.handlers:
                # configurar handler básico solo si no existe configuración previa
                handler = logging.StreamHandler()
                formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.setLevel(logging.INFO)
        else:
            self._logger = logger

    # ----- Propiedades públicas -----
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

    # ----- Métodos principales -----
    def cargar_csv(
        self,
        nombre_archivo: str,
        encoding: str = "utf-8",
        delimiter: str = ",",
        decimal: str = ".",
        usecols: Optional[List[str]] = None,
        parse_dates: Optional[List[str]] = None,
        dtype: Optional[Dict[str, Any]] = None,
        na_values: Optional[List[str]] = None,
        chunksize: Optional[int] = None,
        low_memory: bool = True,
    ) -> pd.DataFrame:
        """
        Carga un CSV en memoria. Si nombre_archivo es una ruta absoluta existente se usa directamente;
        si no, se busca dentro de self._ruta_base.

        :param nombre_archivo: nombre del archivo o ruta
        :param chunksize: si se establece, lee por trozos y concatena (útil para archivos grandes)
        :return: DataFrame cargado
        """
        # Determinar ruta completa: si se pasa una ruta absoluta o relativa que existe, usarla
        posible_path = Path(nombre_archivo)
        if posible_path.exists():
            ruta_completa = posible_path
        else:
            ruta_completa = self._ruta_base / nombre_archivo

        if not ruta_completa.exists():
            msg = f"Archivo no encontrado: {ruta_completa}"
            if self.verbose:
                print(msg)
            self._logger.error(msg)
            raise FileNotFoundError(msg)

        if self.verbose:
            print(f"Iniciando carga de: {ruta_completa}")

        try:
            if chunksize and chunksize > 0:
                # Lectura por chunks para archivos grandes
                reader = pd.read_csv(
                    ruta_completa,
                    encoding=encoding,
                    delimiter=delimiter,
                    decimal=decimal,
                    usecols=usecols,
                    parse_dates=parse_dates,
                    dtype=dtype,
                    na_values=na_values,
                    chunksize=chunksize,
                    low_memory=low_memory
                )
                partes = []
                total_rows = 0
                for i, chunk in enumerate(reader, start=1):
                    partes.append(chunk)
                    total_rows += len(chunk)
                    if self.verbose:
                        print(f"Leído chunk {i}, filas acumuladas: {total_rows}")
                df = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
            else:
                df = pd.read_csv(
                    ruta_completa,
                    encoding=encoding,
                    delimiter=delimiter,
                    decimal=decimal,
                    usecols=usecols,
                    parse_dates=parse_dates,
                    dtype=dtype,
                    na_values=na_values,
                    low_memory=low_memory
                )

            if df.empty:
                msg = f"El archivo {ruta_completa} está vacío"
                self._logger.error(msg)
                raise ValueError(msg)

            self._dataframe = df
            self._nombre_archivo = str(ruta_completa)
            self.calcular_metricas()
            if self.verbose:
                self.registrar_info_carga()
            return self._dataframe

        except pd.errors.EmptyDataError as e:
            self._logger.error(f"El archivo {ruta_completa} está vacío o mal formado: {e}")
            raise
        except Exception as e:
            self._logger.error(f"Error al cargar {ruta_completa}: {e}")
            raise

    def calcular_metricas(self) -> None:
        """Calcula y actualiza num_filas, num_columnas y porcentaje_nulos."""
        if self._dataframe is None:
            self._num_filas = 0
            self._num_columnas = 0
            self._porcentaje_nulos = 0.0
            return

        self._num_filas = int(len(self._dataframe))
        self._num_columnas = int(len(self._dataframe.columns))
        total_valores = self._dataframe.size
        total_nulos = int(self._dataframe.isnull().sum().sum())
        self._porcentaje_nulos = (total_nulos / total_valores) * 100 if total_valores > 0 else 0.0

    def registrar_info_carga(self) -> None:
        """Muestra por consola un resumen amigable de la carga (solo si verbose=True)."""
        if not self.verbose:
            return

        print("=" * 60)
        print("PROYECTO PROGRAMACIÓN 2: INFORMACION ARCHIVO CARGADO")
        print("=" * 60)
        print(f"Archivo: {self._nombre_archivo}")
        print(f"Número de filas: {self._num_filas:,}")
        print(f"Número de columnas: {self._num_columnas}")
        print(f"Porcentaje global de nulos: {self._porcentaje_nulos:.2f}%")
        print("-" * 60)
        print(f"Columnas y tipos:\n{self._dataframe.dtypes}")
        print("-" * 60)
        print(f"Primeras 5 filas: \n{self._dataframe.head(5)}")
        print("-" * 60)

    # ----- Funcionalidades adicionales -----
    def resumen(self) -> Dict[str, Any]:
        """
        Devuelve un resumen con las métricas principales y un pequeño sample.
        """
        return {
            "archivo": self._nombre_archivo,
            "filas": self._num_filas,
            "columnas": self._num_columnas,
            "porcentaje_nulos": round(self._porcentaje_nulos, 2),
            "columnas_tipos": self._dataframe.dtypes.to_dict() if self._dataframe is not None else {},
            "head": self._dataframe.head(5).to_dict(orient="records") if self._dataframe is not None else []
        }


    def filtrar_por_provincia(
        self,
        columna_provincia: str,
        valor_provincia: str,
        in_place: bool = True,
        case_sensitive: bool = False
    ) -> pd.DataFrame:
        """
        Filtra el dataframe por el valor de la columna provincia/cantón.
        :param columna_provincia: nombre de la columna que contiene la provincia/cantón
        :param valor_provincia: valor a filtrar (ej. "Cartago")
        :param in_place: si True actualiza self._dataframe
        :param case_sensitive: si False hace comparación en minúsculas
        :return: DataFrame filtrado
        """
        if self._dataframe is None:
            raise ValueError("No hay dataframe cargado para filtrar.")

        if columna_provincia not in self._dataframe.columns:
            raise KeyError(f"La columna '{columna_provincia}' no existe en el dataframe.")

        if not case_sensitive:
            mask = self._dataframe[columna_provincia].astype(str).str.lower() == str(valor_provincia).lower()
        else:
            mask = self._dataframe[columna_provincia].astype(str) == str(valor_provincia)

        df_filtrado = self._dataframe.loc[mask].copy()

        cliente = ClienteAPI(base_url="https://api.11holidays.com")

        df_api = cliente.obtener_datos(
            endpoint="/v1/holidays",
            params={"country": "CR"}  # parametros
        )

        df_filtrado = self.feriados_por_mes(df_filtrado, df_api)

        df_filtrado = df_filtrado.dropna()
        if in_place:
            self._dataframe = df_filtrado
            self.calcular_metricas()
        return df_filtrado

    def guardar_dataframe_procesado(self, ruta_destino: str, index: bool = False, sep: str = ",") -> str:
        """
        Guarda el dataframe actual en la ruta indicada. Crea carpetas si no existen.
        Devuelve la ruta absoluta del archivo guardado.
        """
        if self._dataframe is None:
            raise ValueError("No hay dataframe cargado para guardar.")

        destino = Path(ruta_destino)
        destino.parent.mkdir(parents=True, exist_ok=True)
        self._dataframe.to_csv(destino, index=index, sep=sep)
        if self.verbose:
            print(f"Guardado dataframe procesado en: {destino.resolve()}")
        return str(destino.resolve())

    # ----- Utilidades -----
    def set_logger(self, logger: logging.Logger) -> None:
        """Permite reemplazar el logger por uno propio (útil para tests)."""
        self._logger = logger

    def feriados_por_mes(self, df_data: pd.DataFrame, df_feriados: pd.DataFrame) -> pd.DataFrame:

        df_feriados["date"] = pd.to_datetime(df_feriados["date"], errors="coerce")
        df_feriados["holiday_month"] = df_feriados["date"].dt.month

        # Agrupar feriados por mes
        df_holidays_month = (
            df_feriados.groupby(["holiday_month"])
            .agg(
                cant_feriados=("name", "count"),
                nombres_feriados=("name", lambda x: list(x))
            )
            .reset_index()
        )

        # Renombrar para que coincida con tu dataset
        df_holidays_month = df_holidays_month.rename(columns={
            "holiday_month": "month"
        })

        # Merge por mes con tus datos locales
        df_final = df_data.merge(df_holidays_month, on=["month"], how="left")

        # Rellenar valores faltantes con defaults
        df_final["cant_feriados"] = df_final["cant_feriados"].fillna(0).astype(int)
        df_final["nombres_feriados"] = df_final["nombres_feriados"].apply(
            lambda x: x if isinstance(x, list) else ["No hay feriados en este mes"]
        )

        # Crear indicador binario si el mes contiene al menos un feriado
        df_final["es_feriado_mes"] = (df_final["cant_feriados"] > 0).astype(int)

        return df_final
# Fin del archivo
