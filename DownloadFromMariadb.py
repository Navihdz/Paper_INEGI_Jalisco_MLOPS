import mariadb
import sys
import pandas as pd

def download():
    # Configuración de la conexión a la base de datos
    config = {
        'host': '127.0.0.1',
        'port': 3307,
        'user': 'root',
        'password': 'root',
        'database': 'database_jalisco'
    }
    # intento de conneccion a MariaDB
    try:
        conn = mariadb.connect(**config)

    except mariadb.Error as e:
        print(f"Error al conectar a maria db: {e}")
        sys.exit(1)

    # Abilitar o desabilitar edicion en DB (Aqunque en este caso no importa)
    conn.autocommit = True

    # creamos conección
    cur = conn.cursor()

    query = """
    SELECT *
    FROM general;"""

    #los datos de respuesta los guardamos en un dataframe de pandas
    df_database = pd.read_sql(query, conn)
    #exportamos csv
    df_database.to_csv("databaseFromSQL.csv")

if __name__ == '__main__':
    download()