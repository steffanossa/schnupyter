import requests
import json
import pandas as pd
my_penguins = pd.read_csv(
    "data/made_up_penguins.csv"
)

my_penguins_as_json = json.dumps(my_penguins)
url = "http://localhost:5000/predict"

response = requests.post(url, json=my_penguins_as_json)
if response.status_code == 200:
    result = response.json()
    prediction = result['prediction']
    print('Vorhersage:', prediction)
else:
    print('Fehler bei der Anfrage.')