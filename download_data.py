# Import required libraries
import requests  # HTTP library for making web requests to download files
import pandas as pd  # Data manipulation library for working with CSV data
import os  # Operating system interface for directory and file operations

# URL of the medical insurance dataset hosted on GitHub
DATA_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv"

# Define the output directory and file path
DATA_DIR = "data"  # Directory name where data will be stored
FILE_PATH = os.path.join(DATA_DIR, "medical_insurance.csv")  # Full path for the saved CSV file

def download_dataset():
    """Download the medical insurance dataset from GitHub and save it locally."""
    
    # Create the data directory if it does not already exist
    # exist_ok=True prevents error if directory already exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print(f"Downloading dataset from:\n{DATA_URL}\n")
    
    # Send an HTTP GET request to the specified URL to fetch the CSV file
    # timeout parameter prevents the request from hanging indefinitely
    response = requests.get(DATA_URL, timeout=30)
    
    # Raise an exception if the HTTP request returned an error status code (4xx or 5xx)
    # This ensures we only proceed if the download was successful
    response.raise_for_status()
    
    # Open the target file in write-binary mode ('wb') and write the response content
    # 'wb' mode is used because we're writing raw bytes from the HTTP response
    with open(FILE_PATH, "wb") as file:
        file.write(response.content)
    
    print(f"Dataset saved to: {FILE_PATH}\n")
    
    # Load the saved CSV file into a pandas DataFrame for inspection
    # This confirms the file was saved correctly and is readable
    df = pd.read_csv(FILE_PATH)
    
    # Print the dimensions of the dataset (rows, columns)
    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns\n")
    
    # Display the first 5 rows to verify the data was loaded correctly
    print("First 5 rows:")
    print(df.head())
    
    return df

# Execute the download function when this script is run directly
# This guard prevents the code from running if the file is imported as a module
if __name__ == "__main__":
    download_dataset()
