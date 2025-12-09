# src/basedatos/gestor_base_datos.py
"""
Gestor de base de datos (SQLite) con separación de responsabilidades.

Clases:
- ConexionSQLite: manejo de la conexión y operaciones SQL básicas.
- IntegradorDatos: preparación del DataFrame y flujo de integración (crear tabla, insertar, crear índices).

Diseñado para funcionar con el DataFrame que produce datos.gestor_datos.GestorDatos.
"""

from pathlib import Path
import sqlite3
import pandas as pd
from typing import Optional, List, Dict, Any, Tuple


class ConexionSQLite:
    """
    Clase responsable de la conexión a SQLite y operaciones de bajo nivel.
    """

    def __init__(self, ruta_db: str = "data/db/demanda_transporte.db"):
        self.ruta_db = Path(ruta_db)
        self.ruta_db.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None

    def conectar(self) -> None:
        """Abre la conexión si no está abierta."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.ruta_db))

    def cerrar(self) -> None:
        """Cierra la conexión si está abierta."""
        if self._conn is not None:
            try:
                self._conn.commit()
            finally:
                self._conn.close()
                self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        """Devuelve la conexión (asegura que esté abierta)."""
        self.conectar()
        assert self._conn is not None
        return self._conn

    def ejecutar(self, sql: str, params: Optional[Tuple] = None) -> None:
        """Ejecuta una sentencia SQL simple."""
        cur = self.conn.cursor()
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)
        self.conn.commit()

    def ejecutar_many(self, sql: str, seq_of_params: List[Tuple]) -> None:
        """Ejecuta executemany y hace commit."""
        cur = self.conn.cursor()
        cur.executemany(sql, seq_of_params)
        self.conn.commit()

    def tabla_existe(self, nombre_tabla: str) -> bool:
        """Comprueba si una tabla existe en la DB."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", (nombre_tabla,)
        )
        return cur.fetchone() is not None


class IntegradorDatos:
    """
    Clase encargada de preparar un DataFrame y cargarlo en SQLite utilizando ConexionSQLite.
    """

    def __init__(self, conexion: Optional[ConexionSQLite] = None):
        self.conexion = conexion or ConexionSQLite()

    # ------------------ Utilidades internas ------------------
    @staticmethod
    def _map_dtype_sqlite(serie: pd.Series) -> str:
        """Mapeo heurístico de pandas dtype a tipo SQLite."""
        dt = serie.dtype
        if pd.api.types.is_integer_dtype(dt) or pd.api.types.is_integer_dtype(pd.Series([1], dtype=dt)):
            return "INTEGER"
        if pd.api.types.is_float_dtype(dt):
            return "REAL"
        # Fechas y strings -> TEXT
        return "TEXT"

    @staticmethod
    def preparar_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Transformaciones mínimas recomendadas:
        - selección de columnas relevantes (si existen)
        - casts seguros a Int64 para id/year/month
        - creación de columna pasajerostotales si es posible
        - strip en strings
        - reemplazo de NaN por None (útil para sqlite)
        """
        cols_sugeridas = [
            "id", "year", "month", "nombre", "codigoruta", "nombreruta",
            "pasajerosadultomayor", "pasajerosregulares"
        ]
        cols_presentes = [c for c in cols_sugeridas if c in df.columns]
        df_proc = df.loc[:, cols_presentes].copy()

        # Casts seguros
        if "id" in df_proc.columns:
            df_proc["id"] = pd.to_numeric(df_proc["id"], errors="coerce").astype("Int64")
        if "year" in df_proc.columns:
            df_proc["year"] = pd.to_numeric(df_proc["year"], errors="coerce").astype("Int64")
        if "month" in df_proc.columns:
            df_proc["month"] = pd.to_numeric(df_proc["month"], errors="coerce").astype("Int64")

        # Crear total de pasajeros si ambas columnas existen
        if ("pasajerosadultomayor" in df_proc.columns) and ("pasajerosregulares" in df_proc.columns):
            df_proc["pasajerostotales"] = (
                pd.to_numeric(df_proc["pasajerosadultomayor"], errors="coerce").fillna(0) +
                pd.to_numeric(df_proc["pasajerosregulares"], errors="coerce").fillna(0)
            )

        # Normalizar strings
        for col in df_proc.select_dtypes(include="object").columns:
            df_proc[col] = df_proc[col].astype(str).str.strip()

        # Reemplazar NaN por None
        df_proc = df_proc.where(pd.notnull(df_proc), None)
        return df_proc

    @staticmethod
    def evaluar_pk(df: pd.DataFrame, candidata: str = "id") -> Tuple[Optional[str], str]:
        """
        Evalúa si la columna candidata puede usarse como PK.
        Retorna (pk_name o None, mensaje).
        """
        if candidata not in df.columns:
            return None, f"No existe la columna '{candidata}'."
        serie = df[candidata]
        if serie.isnull().any():
            return None, f"La columna '{candidata}' contiene valores nulos; no es apta como PK."
        if not serie.is_unique:
            return None, f"La columna '{candidata}' no es única; no es apta como PK."
        return candidata, f"Se usará '{candidata}' como PRIMARY KEY."

    # ------------------ Operaciones DB ------------------
    def crear_tabla_desde_dataframe(self, df: pd.DataFrame, nombre_tabla: str, pk: Optional[str] = "id",
                                    columnas_extra: Optional[Dict[str, str]] = None) -> None:
        """
        Crea una tabla en SQLite con esquema inferido del DataFrame.
        Si pk es None no se define PRIMARY KEY.
        """
        if df is None or df.empty:
            raise ValueError("DataFrame vacío o None")

        columnas_extra = columnas_extra or {}
        cols_sql = []
        for col in df.columns:
            tipo = columnas_extra.get(col) if col in columnas_extra else self._map_dtype_sqlite(df[col])
            if pk and col == pk:
                cols_sql.append(f'"{col}" {tipo} PRIMARY KEY')
            else:
                cols_sql.append(f'"{col}" {tipo}')

        sql = f'CREATE TABLE IF NOT EXISTS "{nombre_tabla}" ({", ".join(cols_sql)});'
        self.conexion.conectar()
        self.conexion.ejecutar(sql)

    def insertar_dataframe(self, df: pd.DataFrame, nombre_tabla: str, pk: Optional[str] = "id",
                           chunk_size: int = 1000) -> int:
        """
        Inserta el DataFrame en la tabla. Si existe PRIMARY KEY en la tabla, INSERT OR REPLACE
        permite hacer upsert. Devuelve el número de filas insertadas/procesadas.
        """
        if df is None or df.empty:
            return 0

        self.conexion.conectar()
        columnas = list(df.columns)
        cols_sql = ", ".join([f'"{c}"' for c in columnas])
        placeholders = ", ".join(["?"] * len(columnas))

        if pk:
            sql = f'INSERT OR REPLACE INTO "{nombre_tabla}" ({cols_sql}) VALUES ({placeholders})'
        else:
            sql = f'INSERT INTO "{nombre_tabla}" ({cols_sql}) VALUES ({placeholders})'

        df_to_insert = df.where(pd.notnull(df), None)

        total = 0
        cur = self.conexion.conn.cursor()
        try:
            for start in range(0, len(df_to_insert), chunk_size):
                chunk = df_to_insert.iloc[start:start + chunk_size]
                tuples = [tuple(row) for row in chunk.to_numpy()]
                try:
                    cur.executemany(sql, tuples)
                    self.conexion.conn.commit()
                    total += len(tuples)
                except Exception:
                    self.conexion.conn.rollback()
                    for t in tuples:
                        try:
                            cur.execute(sql, t)
                        except Exception:
                            continue
                    self.conexion.conn.commit()
                    total += len(tuples)
        finally:
            cur.close()
        return total

    def crear_indice(self, nombre_tabla: str, columnas: List[str], nombre_indice: Optional[str] = None) -> None:
        """
        Crea un índice sobre la tabla para las columnas indicadas.
        """
        if not columnas:
            return
        self.conexion.conectar()
        nombre_indice = nombre_indice or f"idx_{nombre_tabla}_{'_'.join(columnas)}"
        cols_sql = ", ".join([f'"{c}"' for c in columnas])
        sql = f'CREATE INDEX IF NOT EXISTS "{nombre_indice}" ON "{nombre_tabla}" ({cols_sql})'
        self.conexion.ejecutar(sql)

    # ------------------ Flujo de integración alto nivel ------------------
    def integrar_dataframe(self, df: pd.DataFrame, nombre_tabla: str,
                            ruta_db: Optional[str] = None, pk_candidata: str = "id",
                            chunk_size: int = 1000) -> int:
        """
        Flujo alto nivel:
        - prepara df
        - evalua pk candidata
        - crea tabla (intentando con PK; si falla, crea sin PK)
        - inserta por chunks
        - crea índices sobre columnas útiles
        Retorna número de filas insertadas/actualizadas.
        """
        if df is None or df.empty:
            return 0

        df_proc = self.preparar_dataframe(df)

        pk, _ = self.evaluar_pk(df_proc, pk_candidata)
        try:
            self.crear_tabla_desde_dataframe(df_proc, nombre_tabla, pk=pk)
        except Exception:
            self.crear_tabla_desde_dataframe(df_proc, nombre_tabla, pk=None)

        inserted = self.insertar_dataframe(df_proc, nombre_tabla, pk=pk, chunk_size=chunk_size)

        idx_cols = [c for c in ("year", "month", "codigoruta", "nombreruta") if c in df_proc.columns]
        if idx_cols:
            self.crear_indice(nombre_tabla, idx_cols)

        return inserted

    # ------------------ NUEVO: subset por filtro (ej. Cartago) ------------------
    def insertar_subconjunto_por_filtro(
        self,
        df: pd.DataFrame,
        columna: str,
        patron: str,
        nombre_tabla: str,
        ruta_csv_guardado: Optional[str] = None,
        regex: bool = False,
        case_sensitive: bool = False,
        pk_candidata: str = "id",
        chunk_size: int = 1000
    ) -> int:
        """
        Filtra el DataFrame por la columna dada (contiene patrón), guarda el subset en CSV opcionalmente,
        y lo inserta en la DB como una tabla independiente (crea tabla e inserta).
        Retorna el número de filas insertadas/actualizadas.

        :param df: DataFrame origen (por ejemplo gestor.dataframe)
        :param columna: nombre de la columna donde buscar (ej. 'nombreruta')
        :param patron: subcadena o regex a buscar (ej. 'cartago' o r'\\bCartago\\b')
        :param nombre_tabla: tabla destino en la DB (ej. 'pasajeros_cartago')
        :param ruta_csv_guardado: si se provee, guarda el subset en esa ruta (CSV)
        :param regex: si True se interpreta `patron` como regex en str.contains
        :param case_sensitive: sensible a mayúsculas si True
        :param pk_candidata: columna candidata a PK (por defecto 'id')
        :param chunk_size: chunk size para inserciones
        """
        if df is None or df.empty:
            return 0

        if columna not in df.columns:
            raise KeyError(f"La columna '{columna}' no existe en el DataFrame.")

        mask = df[columna].astype(str).str.contains(patron, case=case_sensitive, na=False, regex=regex)
        df_subset = df.loc[mask].copy()

        if ruta_csv_guardado:
            dest = Path(ruta_csv_guardado)
            dest.parent.mkdir(parents=True, exist_ok=True)
            df_subset.to_csv(dest, index=False)

        if df_subset.empty:
            return 0

        df_proc = self.preparar_dataframe(df_subset)

        pk, _ = self.evaluar_pk(df_proc, pk_candidata)
        try:
            self.crear_tabla_desde_dataframe(df_proc, nombre_tabla, pk=pk)
        except Exception:
            self.crear_tabla_desde_dataframe(df_proc, nombre_tabla, pk=None)

        inserted = self.insertar_dataframe(df_proc, nombre_tabla, pk=pk, chunk_size=chunk_size)

        idx_cols = [c for c in ("year", "month", "codigoruta", "nombreruta") if c in df_proc.columns]
        if idx_cols:
            self.crear_indice(nombre_tabla, idx_cols)

        return inserted


# ------------------ Ejemplo de uso (comentado) ------------------
"""
Ejemplo rápido desde main.py:

from datos.gestor_datos import GestorDatos
from basedatos.gestor_base_datos import ConexionSQLite, IntegradorDatos

gestor = GestorDatos(ruta_base="", verbose=False)
gestor.cargar_csv("C:/ruta/absoluta/datos.csv")

conexion = ConexionSQLite(ruta_db="C:/ruta/absoluta/demanda_transporte.db")
integrador = IntegradorDatos(conexion=conexion)

# Tabla completa
integrador.integrar_dataframe(gestor.dataframe, nombre_tabla="pasajeros", chunk_size=500)

# Solo Cartago (filtra por 'nombreruta' que contenga 'cartago')
integrador.insertar_subconjunto_por_filtro(
    df=gestor.dataframe,
    columna="nombreruta",
    patron="cartago",
    nombre_tabla="pasajeros_cartago",
    ruta_csv_guardado="C:/ruta/absoluta/cartago.csv",
    regex=False,
    case_sensitive=False,
    pk_candidata="id",
    chunk_size=500
)

conexion.cerrar()
"""
