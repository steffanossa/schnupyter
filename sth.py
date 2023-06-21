import pandas as pd
from names_dataset import NameDataset

nd = NameDataset()

fems = nd.get_top_names(n=344, gender="Female", country_alpha2="DE")["DE"]["F"]

males = nd.get_top_names(n=344, gender="Male", country_alpha2="DE")["DE"]["M"]

df = pd.read_csv(
    "data/penguins.csv",
    index_col=0
)


df["name"] = ""
i_fem = 0
i_mal = 0
for index, row in df.iterrows():
    if row["sex"] == "female":
        df.at[index, 'name'] = fems[i_fem]
        i_fem += 1
    elif row["sex"] == "male":
        df.at[index, 'name'] = males[i_mal]
        i_mal += 1

print(df)
df.to_csv("data/output.csv", index=False)