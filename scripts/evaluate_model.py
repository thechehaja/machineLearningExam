from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluiraj performanse modela.
    :param model: Trenirani model
    :param X_test: Testni podaci
    :param y_test: Stvarne vrijednosti za testne podatke
    :return: Dictionary s metriksima performansi
    """
    y_pred = model.predict(X_test)

    performance = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    return performance