import requests
import pandas as pd

class ClienteAPI:
    """
    consume API de feriados en cr
    """

    def __init__(self, base_url: str):
        self.base_url = base_url

    def obtener_datos(self, endpoint: str, params: dict = None):
        """
        petecion de tipo GET al endpoint y retorna un DF.
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()  # error si el codigo no es 200
            data = response.json()

            # se espera un formato JSON
            if isinstance(data, list):
                return pd.DataFrame(data)

        except requests.exceptions.RequestException as e:
            print(f"Error al consultar: {e}")
            return pd.DataFrame()
