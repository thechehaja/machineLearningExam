import pandas as pd

def load_data(filepath):
    """
    Učitaj podatke iz Excel datoteke.
    :param filepath: Putanja do Excel datoteke
    :return: DataFrame s učitanim podacima
    """
    df = pd.read_excel(filepath)
    return df
