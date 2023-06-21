#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import panel as pn
pn.extension('tabulator')
import hvplot.pandas
from panel.interact import interact


# In[ ]:


#get_ipython().run_cell_magic('html', '', '<div class="penguin_container">\n    <div class="peng">Pinguine</div>\n    <img src="https://i.natgeofe.com/k/d3ea00a0-773b-437e-b7c6-a32f270b1b5a/adelie-penguin-jumping-ocean.jpg">\n</div>\n<style>\n.penguin_container {\n  position: relative;\n  text-align: center;\n  color: white;\n}\n.peng {\n    text-align: center;\n    \n    position: absolute;\n    color: yellow;\n    font-size: 200px;\n    top: 50%;\n    left: 10%;\n}\n.peng:hover{\n    animation: manimation 0.5s infinite;\n}\n@keyframes manimation {\n  0%   {\n        transform:rotate(5deg)}\n  100% {\n       transform:rotate(0deg)}\n}</style>\n')


# In[ ]:


df = pd.read_csv(
    "data/penguins_wNames.csv"
)


# In[ ]:


df.info()


# In[ ]:


# NAs across the dataset
df.apply(lambda x: sum(x.isnull().values), axis = 0)


# In[ ]:


df = df.dropna()


# In[ ]:


df.hvplot.scatter(x='bill_length_mm', y='bill_depth_mm', by='species')


# In[ ]:


df_nums = df.select_dtypes(include = np.number)
df_corr= df_nums.dropna().corr()
corrplot = sns.heatmap(
    df_corr,
    annot = True
)


# In[ ]:


idf = df.interactive()


# In[ ]:


radiobuttons = {}
for item in ("sex", "island", "species", "year"):
    radiobuttons[item] = pn.widgets.RadioButtonGroup(
        name = "Y axis",
        options = df[item].unique().tolist(),
        button_type = "primary"
    )


# In[ ]:


radiobuttons["island"]


# In[ ]:


# imports
import pandas as pd
import panel as pn
pn.extension('tabulator')
import hvplot.pandas
from panel.interact import interact

# options
islands = ["all"] + df["island"].unique().tolist()
sexes = ["all"] + df["sex"].unique().tolist()

# widgets
sex_select = pn.widgets.Select(
    options=sexes,
    name="Select Sex"
)
island_select = pn.widgets.RadioButtonGroup(
    options=islands,
    name="Select Island",
    button_type = "primary"
)

def create_plot(island, sex):
    
    if island == "all" and sex == "all":
        filtered_df = df.copy()
        
    elif island == "all":
        filtered_df = df[df["sex"] == sex]
        
    elif sex == "all":
        filtered_df = df[df["island"] == island]
        
    else:
        filtered_df = df[(df["island"] == island) & (df["sex"] == sex)]

    return filtered_df.hvplot.scatter(
        y="body_mass_g",
        x="flipper_length_mm",
        by="species",
        hover_cols="name"
    )

interact(
    create_plot,
    island=island_select,
    sex=sex_select
)


# In[ ]:


# imports
import pandas as pd
import panel as pn
pn.extension('tabulator')
import hvplot.pandas
from panel.interact import interact


# options
dimensions = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
groups = ["sex", "year", "island", "species"]


# widgets
y_axis_select = pn.widgets.Select(
    options = dimensions,
    name = "Y-Achse",
    value = dimensions[0]
)

x_axis_select = pn.widgets.Select(
    options = dimensions,
    name = "X-Achse",
    value = dimensions[1]
)

group_select = pn.widgets.Select(
    options = groups,
    name = "Gruppieren nach")


def create_plot(x_axis, y_axis, group_by, data = df):
    return df.hvplot.scatter(
        y = y_axis,
        x = x_axis,
        by = group_by,
        hover_cols = "name"
    )


interact(
    create_plot,
    x_axis = x_axis_select,
    y_axis = y_axis_select,
    group_by = group_select,
)


# ## Splitting into training and test sets

# In[ ]:


from sklearn.model_selection import train_test_split
from pandas import get_dummies

y = df["species"].values
penguins_x = pd.get_dummies(df.drop("species", axis = 1))
x = penguins_x.values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .1, random_state = 187)


# ## Defining a model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier

steps = [("rf", RandomForestClassifier(n_estimators = 100))]


# ## Setting up a pipeline

# In[ ]:


from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps = steps)

rf_pipeline_fit = pipeline.fit(x_train, y_train)

rf_pred_tbl = pd.DataFrame(
{"true": y_test,
"pred": rf_pipeline_fit.predict(x_test)})

rf_pred_tbl


# <h2>Evaluation</h2>

# In[ ]:


from sklearn.metrics import classification_report

classification_report = classification_report(
y_true = rf_pred_tbl.true,
y_pred = rf_pred_tbl.pred)

print(classification_report)


# In[ ]:


from flask import Flask, request, jsonify
model = pipeline.named_steps['rf']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Empfangen Sie die Daten
    # Verwenden Sie Ihr Modell zur Vorhersage
    prediction = model.predict(data)
    return jsonify({'prediction': prediction})  # Senden Sie die Vorhersage als JSON zurück

#app.run()


# In[ ]:

print("galileo")
import requests
import json
my_penguins = pd.read_csv(
    "data/made_up_penguins.csv"
)

# my_penguins_as_json = my_penguins.to_json(orient='records')
# url = "http://localhost:5000/predict"

# response = requests.post(url, json=my_penguins_as_json)
# if response.status_code == 200:
#     result = response.json()
#     prediction = result['prediction']
#     print('Vorhersage:', prediction)
# else:
#     print('Fehler bei der Anfrage.')

# In[ ]:

my_penguins_prediction = model.predict(my_penguins)
print(my_penguins_prediction)