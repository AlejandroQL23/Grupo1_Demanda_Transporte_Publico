from src.api.cliente_api import ClienteAPI


def main():
    cliente = ClienteAPI(base_url="https://api.11holidays.com")

    df = cliente.obtener_datos(
        endpoint="/v1/holidays",
        params={"country": "CR"}  # parametros
    )

    print(df)

if __name__ == "__main__":
    main()
