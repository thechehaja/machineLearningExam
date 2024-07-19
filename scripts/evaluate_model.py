from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test, label_encoders):
    """
    Evaluiraj performanse modela.
    :param model: Trenirani model
    :param X_test: Testni podaci
    :param y_test: Stvarne vrijednosti za testne podatke
    :param label_encoders: Dictionary s enkoderima
    :return: Dictionary s metriksima performansi
    """
    y_pred = model.predict(X_test)

    # Dekodiraj predikcije i stvarne vrijednosti
    target_encoder = label_encoders['dcNyha']
    y_pred = target_encoder.inverse_transform(y_pred)
    y_test = target_encoder.inverse_transform(y_test)

    performance = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=1),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=1)
    }

    return performance
