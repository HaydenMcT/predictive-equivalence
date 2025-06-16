# %%
import numpy as np 
import pandas as pd 



# %%
first = True
for i in range(48): 
    # if rf_i exists, then read it and append to the dataframe
    try:
        rf_i = pd.read_csv(str(i)+'.csv')

    except:
        print(f'result {i} missing')
        continue
    if first:
        rf = rf_i
        first = False
    else:
        rf = rf._append(rf_i)
# merge the dataframes
rf.to_csv('rset.csv', index=False)


# %%



