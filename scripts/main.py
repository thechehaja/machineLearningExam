from load_data import load_data
from clean_data import clean_data
from train_model import train_model
from evaluate_model import evaluate_model

def main():
    # Putanja do dataset-a
    data_path = 'data/kardiologija_hospitalizacija.xlsx'
    model_path = 'models/random_forest.pkl'

    # Učitaj podatke
    df = load_data(data_path)

    # Očisti podatke
    df = clean_data(df)

    # Treniraj model
    target_column = 'dcNyha'  # Promijeni na željenu ciljnu kolonu
    model, X_test, y_test, label_encoders = train_model(df, target_column, model_path)

    # Evaluiraj model
    performance = evaluate_model(model, X_test, y_test, label_encoders)

    # Prikaži performanse
    for metric, value in performance.items():
        print(f'{metric}: {value:.4f}')

if __name__ == "__main__":
    main()