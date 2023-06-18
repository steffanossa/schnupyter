```python
import pandas as pd
import seaborn as sns
import numpy as np
import panel as pn
pn.extension('tabulator')
import hvplot.pandas
from panel.interact import interact
```

```python
df = pd.read_csv(
    "data/penguins_wNames.csv"
)
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 333 entries, 0 to 332
    Data columns (total 9 columns):
     #   Column             Non-Null Count  Dtype  
    ---  ------             --------------  -----  
     0   species            333 non-null    object 
     1   island             333 non-null    object 
     2   bill_length_mm     333 non-null    float64
     3   bill_depth_mm      333 non-null    float64
     4   flipper_length_mm  333 non-null    float64
     5   body_mass_g        333 non-null    float64
     6   sex                333 non-null    object 
     7   year               333 non-null    int64  
     8   name               333 non-null    object 
    dtypes: float64(4), int64(1), object(4)
    memory usage: 23.5+ KB
    


```python
# NAs across the dataset
df.apply(lambda x: sum(x.isnull().values), axis = 0)
```




    species              0
    island               0
    bill_length_mm       0
    bill_depth_mm        0
    flipper_length_mm    0
    body_mass_g          0
    sex                  0
    year                 0
    name                 0
    dtype: int32




```python
df = df.dropna()
```


```python
df.hvplot.scatter(x='bill_length_mm', y='bill_depth_mm', by='species')
```


```python
df_nums = df.select_dtypes(include=np.number)
df_corr= df_nums.dropna().corr()
corrplot = sns.heatmap(
    df_corr,
    annot = False,
)
```


    
![png](output_6_0.png)
    



```python
idf = df.interactive()
```


```python
radiobuttons = {}
for item in ("sex", "island", "species", "year"):
    radiobuttons[item] = pn.widgets.RadioButtonGroup(
        name = "Y axis",
        options = list(df[item].unique()),
        button_type = "primary"
    )
```


```python
radiobuttons["species"]
```


```python
islands = df["island"].unique().tolist()

island_select = pn.widgets.Select(name="Select Island", options=islands)
island_select = pn.widgets.RadioButtonGroup(name="Select Island", options = islands, button_type = "primary")

def create_plot(island):
    return df[df["island"] == island].hvplot(y = "body_mass_g",
                                             x = "flipper_length_mm",
                                             kind = "scatter",
                                             by = "species"
                                            )

interact(create_plot, island=island_select)
```


```python
islands = df["island"].unique().tolist()
sexes = df["sex"].unique().tolist()

sex_select = pn.widgets.Select(options=sexes)
island_select = pn.widgets.RadioButtonGroup(options=islands)

def create_plot(island, sex):
    return df[(df["island"] == island) & (df["sex"] == sex)].hvplot.scatter(
        y="body_mass_g",
        x="flipper_length_mm",
        by="species",
        hover_cols = "name")

interact(create_plot, island=island_select, sex=sex_select)
```

## Splitting into training and test sets


```python
from sklearn.model_selection import train_test_split
from pandas import get_dummies

y = df["species"].values
penguins_x = pd.get_dummies(df.drop("species", axis = 1))
x = penguins_x.values

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = .1, random_state = 187)
```

## Defining a model


```python
from sklearn.ensemble import RandomForestClassifier

steps = [("rf", RandomForestClassifier(n_estimators = 100))]
```

## Setting up a pipeline


```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline(steps = steps)

rf_pipeline_model = pipeline.fit(x_train, y_train)

rf_pred_tbl = pd.DataFrame(
{"true": y_test,
"pred": rf_pipeline_model.predict(x_test)})

rf_pred_tbl
```





<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>true</th>
      <th>pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelie</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Chinstrap</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Adelie</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Gentoo</td>
      <td>Gentoo</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Gentoo</td>
      <td>Gentoo</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>295</th>
      <td>Chinstrap</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Adelie</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>297</th>
      <td>Chinstrap</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>298</th>
      <td>Adelie</td>
      <td>Adelie</td>
    </tr>
    <tr>
      <th>299</th>
      <td>Chinstrap</td>
      <td>Chinstrap</td>
    </tr>
  </tbody>
</table>
<p>300 rows × 2 columns</p>
</div>



<h2>Evaluation</h2>


```python
from sklearn.metrics import classification_report

classification_report = classification_report(
y_true = rf_pred_tbl.true,
y_pred = rf_pred_tbl.pred)

print(classification_report)
```

                  precision    recall  f1-score   support
    
          Adelie       0.72      1.00      0.84       126
       Chinstrap       1.00      0.23      0.38        64
          Gentoo       1.00      1.00      1.00       110
    
        accuracy                           0.84       300
       macro avg       0.91      0.74      0.74       300
    weighted avg       0.88      0.84      0.80       300
    
    


```python

```
