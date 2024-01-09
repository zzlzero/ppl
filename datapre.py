import pandas as pd
import numpy as np
import re
df = pd.read_csv("so_data.csv") 
data = df['Title']
text_data = open('so_data.txt', 'w')
for item in data:

    text_data.write(item)
    text_data.write('\n')
text_data.close()