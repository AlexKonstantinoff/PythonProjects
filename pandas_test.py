#%%
from IPython.display import display
import pandas as pd
import mglearn

data = {
'Name': ['John', 'Anna', 'Peter', 'Linda'],
'Location': ["New York", "Paris", "Berlin", "London"],
'Age' : [24, 13, 53, 33]    
}

data_p = pd.DataFrame(data)
display(data_p[data_p.Age > 20])

# %%