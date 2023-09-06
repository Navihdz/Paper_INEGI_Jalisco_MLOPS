from sqlalchemy import create_engine
import pandas as pd

def upload():
    # Configuración de la conexión a la base de datos
    config = {
        'host': '127.0.0.1',
        'port': 3307,
        'user': 'root',
        'password': 'root',
        'database': 'database_jalisco'
    }

    # Carga los datos desde el archivo CSV en un DataFrame
    df = pd.read_csv('DataBaseToUpload_SQL.csv')

    # Crea una conexión a la base de datos utilizando el driver mysqlconnector
    #engine = create_engine(f"mysql+mysqlconnector://{config['user']}:{config['password']}@{config['host']}:{config['port']}/{config['database']}")
    engine = create_engine("mysql://root:root@127.0.0.1:3307/database_jalisco")

    # Carga el DataFrame en la tabla correspondiente
    df.to_sql('general', con=engine, if_exists='replace', index=False)

if __name__ == '__main__':
    upload()