import pandas as pd
import numpy as np
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler

a = list(range(2,43))
a.remove(10)
a.remove(24)
a.remove(38)

for i in a:
    df = pd.read_csv('/Test_input/ME'+ str(i) +'_add_motif.csv',index_col=0,header=0)
    data = df.T
    sc = StandardScaler()
    data = sc.fit_transform(data) 
    models = Path('/RF_model_save/ME'+ str(i) +'/').glob('*.pkl')
    for model in models:
        (filepath,tempfilename) = os.path.split(model)
        (filename,extension) = os.path.splitext(tempfilename)
        print(filename)
        xlf_lo = joblib.load(model)
        y_pred = xlf_lo.predict(data)
        y_pred = DataFrame(y_pred).T
        dir = os.getcwd()
        os.mkdir(dir+'/Test_output/')
        with open('/Test_output/ME'+ str(i) +'_pre.csv','a') as f:
            f.write('')
        y_pred.to_csv('/Test_output/ME'+ str(i) +'_pre.csv', 
                      index = False, header = False, mode='a')
            
print('Scm6A done')
