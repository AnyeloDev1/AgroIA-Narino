import os
import pandas as pd
from sodapy import Socrata
from dotenv import load_dotenv

load_dotenv()

class SocrataConnector:
    """Conector base para extraer Datos Abiertos del gobierno colombiano."""
    
    def __init__(self):
        # Usamos un token de datos.gov.co opcional si está en el archivo .env
        self.token = os.getenv("SOCRATA_APP_TOKEN", None)
        self.client = Socrata("www.datos.gov.co", self.token)

    def fetch_dataset(self, dataset_id: str, limit: int = 500, where_clause: str = None) -> pd.DataFrame:
        """Recupera un conjunto de datos aplicando filtros del lado del servidor."""
        try:
            results = self.client.get(dataset_id, limit=limit, where=where_clause)
            if not results:
                return pd.DataFrame()
            return pd.DataFrame.from_records(results)
        except Exception as e:
            print(f"Error al conectar con el dataset {dataset_id}: {e}")
            return pd.DataFrame()