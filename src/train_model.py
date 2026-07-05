# Import required libraries
import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.linear_model import LinearRegression  # For Linear Regression model
from sklearn.tree import DecisionTreeRegressor  # For Decision Tree model
from sklearn.ensemble import RandomForestRegressor  # For Random Forest model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # For evaluation metrics
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # For preprocessing categorical and numerical data
from sklearn.compose import ColumnTransformer  # For applying different transformations to different columns
from sklearn.pipeline import Pipeline  # For creating ML pipelines
import joblib  # For saving and loading models
import os  # For file system operations
import json  # For JSON file handling


def load_data(path):
    """
    Load data from a CSV file and return a DataFrame.
    """
    df = pd.read_csv(path)  # Read the CSV file into a pandas DataFrame
    return df  # Return the loaded DataFrame


def preprocess_data(df):
    """
    Preprocess the data by separating features and target,
    and creating a ColumnTransformer for different column types.
    """
    df_copy = df.copy()  # Create a copy of the dataframe to avoid modifying original

    X = df_copy.drop('charges', axis=1)  # Separate features (all columns except 'charges')
    y = df_copy['charges']  # Separate target variable (charges column)

    categorical_cols = ['sex', 'smoker', 'region']  # List of categorical columns to encode
    numerical_cols = ['age', 'bmi', 'children']  # List of numerical columns to scale

    categorical_transformer = OneHotEncoder(drop='first', sparse_output=False)  # Create OneHotEncoder for categorical columns, drop first to avoid multicollinearity

    numerical_transformer = StandardScaler()  # Create StandardScaler for numerical columns

    preprocessor = ColumnTransformer(  # Create ColumnTransformer to apply different transformations
        transformers=[
            ('num', numerical_transformer, numerical_cols),  # Apply StandardScaler to numerical columns
            ('cat', categorical_transformer, categorical_cols)  # Apply OneHotEncoder to categorical columns
        ]
    )

    X_transformed = preprocessor.fit_transform(X)  # Fit and transform the features

    return preprocessor, X_transformed, y  # Return preprocessor, transformed features, and target


def train_models(X_train, y_train):
    """
    Train multiple ML models on the training data.
    """
    models = {}  # Initialize empty dictionary to store trained models

    lr_model = LinearRegression()  # Create Linear Regression model instance
    lr_model.fit(X_train, y_train)  # Train Linear Regression model on training data
    models['Linear Regression'] = lr_model  # Store trained model in dictionary

    dt_model = DecisionTreeRegressor(random_state=42)  # Create Decision Tree model instance with fixed random state
    dt_model.fit(X_train, y_train)  # Train Decision Tree model on training data
    models['Decision Tree'] = dt_model  # Store trained model in dictionary

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # Create Random Forest model with 100 trees
    rf_model.fit(X_train, y_train)  # Train Random Forest model on training data
    models['Random Forest'] = rf_model  # Store trained model in dictionary

    return models  # Return dictionary of trained models


def evaluate_models(models, X_test, y_test):
    """
    Evaluate each trained model using multiple regression metrics.
    """
    results = {}  # Initialize empty dictionary to store evaluation results

    for name, model in models.items():  # Iterate through each trained model
        y_pred = model.predict(X_test)  # Generate predictions on test data

        mae = mean_absolute_error(y_test, y_pred)  # Calculate Mean Absolute Error
        mse = mean_squared_error(y_test, y_pred)  # Calculate Mean Squared Error
        rmse = np.sqrt(mse)  # Calculate Root Mean Squared Error
        r2 = r2_score(y_test, y_pred)  # Calculate R-squared score

        results[name] = {  # Store all metrics for this model
            'MAE': mae,  # Mean Absolute Error
            'MSE': mse,  # Mean Squared Error
            'RMSE': rmse,  # Root Mean Squared Error
            'R2': r2  # R-squared score
        }

    return results  # Return dictionary of evaluation results


def main():
    """
    Main function to orchestrate the entire ML pipeline.
    """
    # Get the directory where this script lives (src/)
    # Then go up one level to the project root, then into data/
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'medical_insurance.csv')  # Full path to input data file

    df = load_data(data_path)  # Load the dataset from CSV file

    preprocessor, X, y = preprocess_data(df)  # Preprocess data and get transformed features

    X_train, X_test, y_train, y_test = train_test_split(  # Split data into training and test sets
        X, y, test_size=0.2, random_state=42  # 80% train, 20% test, fixed random state for reproducibility
    )

    models = train_models(X_train, y_train)  # Train all three ML models on training data

    results = evaluate_models(models, X_test, y_test)  # Evaluate all models on test data

    best_model_name = max(results, key=lambda x: results[x]['R2'])  # Find model with highest R2 score
    best_model = models[best_model_name]  # Get the best trained model object

    artifacts_dir = os.path.join(base_dir, 'artifacts')  # Full path to artifacts directory

    os.makedirs(artifacts_dir, exist_ok=True)  # Create artifacts directory if it doesn't exist

    joblib.dump(best_model, os.path.join(artifacts_dir, 'model.pkl'))  # Save best trained model to disk

    joblib.dump(preprocessor, os.path.join(artifacts_dir, 'preprocessor.pkl'))  # Save preprocessor to disk

    with open(os.path.join(artifacts_dir, 'metrics.json'), 'w') as f:  # Open metrics file for writing
        json.dump(results, f, indent=4)  # Save evaluation metrics to JSON file

    print("\nModel Comparison Results:")  # Print header for comparison table
    print("-" * 60)  # Print separator line
    for name, metrics in results.items():  # Iterate through each model's results
        print(f"\n{name}:")  # Print model name
        print(f"  MAE:  ${metrics['MAE']:.2f}")  # Print Mean Absolute Error formatted as currency
        print(f"  MSE:  ${metrics['MSE']:.2f}")  # Print Mean Squared Error formatted as currency
        print(f"  RMSE: ${metrics['RMSE']:.2f}")  # Print Root Mean Squared Error formatted as currency
        print(f"  R2:   {metrics['R2']:.4f}")  # Print R-squared score with 4 decimal places
    print("-" * 60)  # Print separator line
    print(f"\nBest Model: {best_model_name} (R2: {results[best_model_name]['R2']:.4f})")  # Print the best model name and its R2 score


if __name__ == '__main__':  # Check if script is being run directly
    main()  # Execute the main function
