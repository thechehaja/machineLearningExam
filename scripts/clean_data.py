def clean_data(df):
    """
    Očisti dataset popunjavanjem nedostajućih vrijednosti.
    :param df: DataFrame s podacima
    :return: Očišćen DataFrame
    """
    # Pretvori sve kolone u stringove
    df = df.applymap(str)

    # Popuni nedostajuće vrijednosti za numeričke kolone sa medianom
    for column in df.select_dtypes(include=['float64', 'int64']).columns:
        df[column] = df[column].fillna(df[column].median())

    # Popuni nedostajuće vrijednosti za kategorijske kolone sa najčešćim vrijednostima (mode)
    for column in df.select_dtypes(include=['object']).columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    return df
