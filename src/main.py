"""
Main de integración usando rutas ABSOLUTAS.
Opciones:
  1) Integrar todos los datos en tabla 'pasajeros'
  2) Integrar solo Cartago en tabla 'pasajeros_cartago' y guardar CSV filtrado
"""

from pathlib import Path

from src.basedatos.gestor_base_datos import ConexionSQLite, IntegradorDatos
from src.datos.gestor_datos import GestorDatos
from src.modelos.modeloml import ModeloML


def main():
    print("=== INTEGRACIÓN: CSV -> SQLITE (RUTAS ABSOLUTAS) ===\n")

    # ---------- 1. RUTA ABSOLUTA DEL CSV ----------
    ruta_csv = input("Ingresa la RUTA ABSOLUTA del CSV (ej: C:/Users/.../datos.csv): ").strip()
    if not ruta_csv:
        print("No se ingresó ruta. Saliendo.")
        return

    ruta_csv_abs = Path(ruta_csv).expanduser().resolve()
    if not ruta_csv_abs.exists():
        print(f"ERROR: El archivo no existe: {ruta_csv_abs}")
        return

    # ---------- 2. CARGAR CSV CON GestorDatos ----------
    # ruta_base="" para que no interfiera con rutas absolutas
    gestor = GestorDatos(ruta_base="", verbose=True)

    try:
        gestor.cargar_csv(str(ruta_csv_abs))
    except Exception as e:
        print(f"ERROR al cargar el CSV: {e}")
        return

    resumen = gestor.resumen()
    print("\nResumen del DataFrame cargado:")
    print(f" Archivo: {resumen.get('archivo')}")
    print(f" Filas: {resumen.get('filas')}")
    print(f" Columnas: {resumen.get('columnas')}")
    print(f" % Nulos: {resumen.get('porcentaje_nulos')}")
    print(" Columnas detectadas:", list(resumen.get("columnas_tipos", {}).keys()))

    # ---------- 3. RUTA ABSOLUTA DE LA BD ----------
    ruta_db = input("\nRuta ABSOLUTA para la DB SQLite (Enter = C:/demanda_transporte.db): ").strip()
    if not ruta_db:
        ruta_db_abs = Path("C:/demanda_transporte.db").resolve()
    else:
        ruta_db_abs = Path(ruta_db).expanduser().resolve()

    print(f"Usando base de datos: {ruta_db_abs}")

    # Crear conexión e integrador
    conexion = ConexionSQLite(ruta_db=str(ruta_db_abs))
    integrador = IntegradorDatos(conexion=conexion)

    # ---------- 4. MENÚ DE INTEGRACIÓN ----------
    while True:
        print("\n=== MENÚ DE INTEGRACIÓN ===")
        print("1) Integrar TODOS los datos en tabla 'pasajeros'")
        print("2) Integrar SOLO Cartago en tabla 'pasajeros_cartago' y guardar CSV filtrado")
        print("3) Modelo Supervisado - Regresión")
        print("4) Modelo Supervisado - Clasificación (Regresión logística) ")
        print("5) Salir")

        opcion = input("Selecciona una opción (1-5): ").strip()

        if opcion == "1":
            # Integrar todo el DataFrame en la tabla 'pasajeros'
            try:
                filas = integrador.integrar_dataframe(
                    gestor.dataframe,
                    nombre_tabla="pasajeros",
                    chunk_size=500
                )
                print(f"\nIntegración COMPLETA terminada. Filas insertadas/actualizadas en 'pasajeros': {filas}")
            except Exception as e:
                print(f"ERROR al integrar todos los datos: {e}")

        elif opcion == "2":
            # Integrar solo Cartago
            # Ruta por defecto para el CSV filtrado
            ruta_csv_cartago = input(
                "Ruta ABSOLUTA para guardar el CSV de Cartago "
                "(Enter = C:/cartago.csv): "
            ).strip()
            if not ruta_csv_cartago:
                ruta_csv_cartago_abs = Path("C:/cartago.csv").resolve()
            else:
                ruta_csv_cartago_abs = Path(ruta_csv_cartago).expanduser().resolve()

            try:
                filas_cartago = integrador.insertar_subconjunto_por_filtro(
                    df=gestor.dataframe,
                    columna="nombreruta",
                    patron="cartago",             # subcadena 'cartago' (no sensible a mayúsculas)
                    nombre_tabla="pasajeros_cartago",
                    ruta_csv_guardado=str(ruta_csv_cartago_abs),
                    regex=False,
                    case_sensitive=False,
                    pk_candidata="id",
                    chunk_size=500
                )
                print(
                    f"\nIntegración Cartago terminada. "
                    f"Filas insertadas/actualizadas en 'pasajeros_cartago': {filas_cartago}"
                )
                print(f"CSV filtrado guardado en: {ruta_csv_cartago_abs}")
            except Exception as e:
                print(f"ERROR al integrar solo Cartago: {e}")

        elif opcion == "3":
            modelo1 = ModeloML()
            modelo1.regresion()
            modelo1.probar_modelo()

        elif opcion == "4":
            modelo2 = ModeloML()
            modelo2.clasificacion()
            modelo2.probar_modelo2()

        elif opcion == "5":
            print("\nSaliendo y cerrando conexión...")
            break
        else:
            print("Opción inválida. Intenta de nuevo.")

    # ---------- 5. CERRAR CONEXIÓN ----------
    try:
        conexion.cerrar()
    except Exception:
        pass

    print(f"\nBase de datos final en: {ruta_db_abs}")
    print("Fin del programa.")


if __name__ == "__main__":
    main()



