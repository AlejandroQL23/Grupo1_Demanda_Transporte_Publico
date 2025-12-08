# src/main.py
from pathlib import Path
from src.datos.gestor_datos import GestorDatos
from src.api.cliente_api import ClienteAPI
from src.modelos.modeloml import ModeloML


def main():

    print("=== GESTOR DE DATOS — PROYECTO (FILTRAR POR nombreruta: Cartago) ===\n")

    # Ajusta ruta_base si tu CSV no está en src (por ejemplo "." o "data/raw")
    gestor = GestorDatos(ruta_base="src", verbose=True)

    while True:
        print("\n1) Cargar CSV (ej: C:\repos\Grupo1_Demanda_Transporte_Publico\data\raw\datos.csv)")
        print("2) Mostrar resumen")
        print("3) Filtrar filas donde 'nombreruta' contenga 'Cartago'")
        print("4) Guardar dataframe procesado (ruta destino ej: C:\repos\Grupo1_Demanda_Transporte_Publico\data\processed\datos_cartago.csv )")
        print("5) Ver primeras 5 filas")
        print("6) Modelo Regresion Lineal")
        print("7) Modelo ")
        print("8) Salir")

        opt = input("Elige opción (1-8): ").strip()

        if opt == "1":
            nombre = input("Nombre o ruta del CSV [datos.csv]: ").strip() or "datos.csv" #ruta absoluta C:\repos\Grupo1_Demanda_Transporte_Publico\data\raw\datos.csv
            try:
                gestor.cargar_csv(nombre_archivo=nombre)
                print("Archivo cargado correctamente.")
            except FileNotFoundError:
                print("ERROR: archivo no encontrado. Revisa ruta_base o poner la ruta absoluta.")
            except Exception as e:
                print(f"ERROR al cargar: {e}")

        elif opt == "2":
            if gestor.dataframe is None:
                print("No hay DataFrame cargado.")
                continue
            r = gestor.resumen()
            print(f"\nArchivo: {r['archivo']}")
            print(f"Filas: {r['filas']}, Columnas: {r['columnas']}, %Nulos: {r['porcentaje_nulos']}")
            print("Columnas:", list(r['columnas_tipos'].keys()))

        elif opt == "3":
            if gestor.dataframe is None:
                print("Carga primero el CSV.")
                continue
            col_name = "nombreruta"
            if col_name not in gestor.dataframe.columns:
                print(f"No existe la columna '{col_name}'. Columnas actuales: {list(gestor.dataframe.columns)}")
                continue
            # Búsqueda case-insensitive por la subcadena 'cartago'
            mask = gestor.dataframe[col_name].astype(str).str.contains("cartago", case=False, na=False)
            df_filtrado = gestor.dataframe.loc[mask].copy()
            gestor._dataframe = df_filtrado  # actualizamos el dataframe interno (simple y directo)
            gestor.calcular_metricas()
            print(f"Filtrado aplicado. Filas actuales: {gestor.num_filas}")
            if gestor.num_filas > 0:
                print("Ejemplo (primeras 5 filas):")
                print(gestor.dataframe.head(5))
            else:
                print("No se encontraron filas con 'Cartago' en 'nombreruta'.")

        elif opt == "4":
            if gestor.dataframe is None:
                print("No hay DataFrame para guardar.")
                continue
            ruta = input("Ruta destino (ej: data/processed/datos_cartago.csv): ").strip()
            if not ruta:
                print("No ingresaste ruta.")
                continue
            try:
                dest = gestor.guardar_dataframe_procesado(ruta, index=False)
                print(f"Guardado en: {dest}")
            except Exception as e:
                print(f"ERROR al guardar: {e}")

        elif opt == "5":
            if gestor.dataframe is None:
                print("No hay DataFrame cargado.")
                continue
            print(gestor.dataframe.head(5))

        elif opt == "6":
            modelo1 = ModeloML()
            df = modelo1.regresion()

        elif opt == "7":
            modelo1 = ModeloML()
            df = modelo1.clasificacion()

        elif opt == "8":
            print("Saliendo. ¡Éxitos!")
            break

        else:
            print("Opción inválida. Intenta de nuevo.")

if __name__ == "__main__":
    main()
