import requests
import pandas as pd

# Make a GET request to the URL of the CSV file
url = 'https://www.kaggle.com/datasets/luddarell/101-simple-linear-regressioncsv'
response = requests.get(url)

# Read the content of the response into a pandas DataFrame
df = pd.read_csv(response.content)
