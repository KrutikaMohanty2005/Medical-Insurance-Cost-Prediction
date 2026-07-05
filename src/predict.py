# Import joblib for loading saved model and preprocessor objects
import joblib

# Import numpy for numerical operations
import numpy as np

# Import pandas for DataFrame creation
import pandas as pd

# Import os for file path handling
import os


# Define the InsurancePredictor class for making predictions
class InsurancePredictor:
    """
    A class to load the trained model and preprocessor,
    and make predictions on new insurance data.
    """

    # Define the constructor that loads model and preprocessor
    def __init__(self):
        # Get the directory of this script (src/), then go up to project root
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # Construct the path to the model artifact file
        model_path = os.path.join(base_dir, 'artifacts', 'model.pkl')
        # Construct the path to the preprocessor artifact file
        preprocessor_path = os.path.join(base_dir, 'artifacts', 'preprocessor.pkl')

        # Load the trained model from the saved file
        self.model = joblib.load(model_path)
        # Load the preprocessor from the saved file
        self.preprocessor = joblib.load(preprocessor_path)

    # Define the predict method that takes individual feature values as input
    def predict(self, age, sex, bmi, children, smoker, region):
        """
        Predict the insurance charge for a given set of input features.

        Parameters:
        -----------
        age : int
            Age of the individual
        sex : str
            Gender of the individual ('male' or 'female')
        bmi : float
            Body Mass Index of the individual
        children : int
            Number of children covered by insurance
        smoker : str
            Smoking status ('yes' or 'no')
        region : str
            Residential region ('northeast', 'northwest', 'southeast', 'southwest')

        Returns:
        --------
        float
            The predicted insurance charge
        """
        # Create a dictionary with the input features as column names
        input_data = {
            'age': [age],
            'sex': [sex],
            'bmi': [bmi],
            'children': [children],
            'smoker': [smoker],
            'region': [region]
        }

        # Convert the dictionary into a pandas DataFrame for processing
        df = pd.DataFrame(input_data)

        # Transform the input DataFrame using the loaded preprocessor
        # This applies the same transformations used during training
        df_transformed = self.preprocessor.transform(df)

        # Use the trained model to make a prediction on the transformed data
        prediction = self.model.predict(df_transformed)

        # Return the predicted charge as a single float value
        return float(prediction[0])


# Define the main function to demonstrate the predictor usage
def main():
    # Print a header message for the demonstration
    print("=" * 50)
    print("Insurance Cost Prediction")
    print("=" * 50)

    # Create an instance of the InsurancePredictor class
    predictor = InsurancePredictor()

    # Define sample input values for prediction
    age = 35
    sex = "male"
    bmi = 28.5
    children = 2
    smoker = "no"
    region = "southeast"

    # Print the input values being used for prediction
    print(f"\nInput Features:")
    print(f"  Age: {age}")
    print(f"  Sex: {sex}")
    print(f"  BMI: {bmi}")
    print(f"  Children: {children}")
    print(f"  Smoker: {smoker}")
    print(f"  Region: {region}")

    # Call the predict method to get the predicted insurance charge
    predicted_charge = predictor.predict(age, sex, bmi, children, smoker, region)

    # Print the predicted insurance charge with proper formatting
    print(f"\nPredicted Insurance Charge: ${predicted_charge:,.2f}")
    print("=" * 50)


# Check if the script is being run directly (not imported)
if __name__ == '__main__':
    # Call the main function to execute the demonstration
    main()
