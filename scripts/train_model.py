from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

def train_model(df, target_column, model_path):
    """
    Treniraj RandomForest model na datasetu i sačuvaj model.
    :param df: DataFrame s podacima
    :param target_column: Ime ciljne kolone
    :param model_path: Putanja za čuvanje modela
    """
    # Pretvori sve kolone u stringove
    df = df.astype(str)

    # Pretvori kategorijske varijable u numeričke, uključujući ciljnu kolonu
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        label_encoders[column] = le

    # Odvajanje značajki (features) i cilja (target)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Podjela podataka na trening i test setove
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treniranje RandomForest modela
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Sačuvaj model
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

    return model, X_test, y_test, label_encoders
