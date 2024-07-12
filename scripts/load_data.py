import pandas as pd

def load_data(filepath):
    """
    Učitaj podatke iz CSV datoteke.
    :param filepath: Putanja do CSV datoteke
    :return: DataFrame s učitanim podacima
    """
    df = pd.read_csv(filepath)
    return df
