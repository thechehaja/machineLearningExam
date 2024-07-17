
# Healthcare Data Classification Project

## Project Description

This project aims to build a machine learning model to classify medical conditions based on patient data. The dataset includes various features such as age, gender, blood type, medical condition, admission details, and more. The goal is to predict the target variable using a RandomForestClassifier and evaluate its performance.

## Key Features of the Project

1. **Data Loading and Cleaning**
   - Loading the healthcare dataset from a CSV file.
   - Cleaning the data by handling missing values and normalizing text data (e.g., patient names).

2. **Feature Engineering**
   - Transforming categorical variables into numerical values.
   - Balancing classes using the SMOTE technique to handle any class imbalances.

3. **Model Training and Optimization**
   - Training a RandomForestClassifier on the dataset.
   - Using Grid Search for hyperparameter optimization to find the best model configuration.
   - Saving the trained model for future use.

4. **Model Evaluation**
   - Evaluating the model using accuracy, precision, recall, and F1-score.
   - Implementing cross-validation to ensure robust performance metrics.

## Installation

### Clone the Repository
```sh
git clone https://github.com/thechehaja/machineLearningExam.git
cd machineLearningExam
```

### Set Up Virtual Environment
Create and activate a virtual environment to manage dependencies.
```sh
python -m venv venv
source venv/bin/activate  # On Windows use \`venv\Scripts\activate\`
```

### Install Dependencies
Install the required libraries using \`pip\`:
```sh
pip install -r requirements.txt
```

## Running the Project

### Run the Main Script
Execute the main script to load data, train the model, and evaluate its performance.
```sh
python scripts/main.py
```

## Project Structure

```
healthcare-data-classification/
├── .idea/
├── venv/
├── data/
│   └── healthcare_dataset.csv
├── scripts/
│   ├── load_data.py
│   ├── clean_data.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── main.py
├── models/
│   └── random_forest.pkl
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.x
- Libraries specified in \`requirements.txt\`

## Future Work

- Experiment with different machine learning algorithms.
- Further feature engineering to improve model performance.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
