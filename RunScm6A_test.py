import pandas as pd
import numpy as np
from pathlib import Path
import os
import joblib
from pandas.core.frame import DataFrame
from sklearn.preprocessing import StandardScaler

a = list(range(2,43))
a.remove(10)
a.remove(24)
a.remove(38)

def runmodels():
    global a
    for i in a:
        df = pd.read_csv('./Test_input/ME'+ str(i) +'_add_motif.csv',index_col=0,header=0)
        data = df.T
        sc = StandardScaler()
        data = sc.fit_transform(data) 
        models = Path('./RF_model_save/ME'+ str(i) +'/').glob('*.pkl')
        for model in models:
            (filepath,tempfilename) = os.path.split(model)
            (filename,extension) = os.path.splitext(tempfilename)
            print(filename)
            xlf_lo = joblib.load(model)
            y_pred = xlf_lo.predict(data)
            y_pred = DataFrame(y_pred).T
            with open('./Test_output/ME'+ str(i) +'_pre.csv','a') as f:
                f.write('')
            y_pred.to_csv('./Test_output/ME'+ str(i) +'_pre.csv', 
                          index = False, header = False, mode='a')

def addC():
    #Adding Column names
    cell = pd.read_csv("./Test_input/ME2_add_motif.csv",header=None)
    cell = cell.head(1)
    del cell[0]
    global a
    MEname = [f'./Test_output/ME{i}_pre.csv' for i in a]
    for i in range(0,len(MEname)):
        MEn = pd.read_csv(MEname[i],header=None,sep=",")
        C = DataFrame(np.append(cell,MEn,axis=0))
        C.to_csv(str(MEname[i]), index = False, header = False, mode='w')
        
def addR():
    #Adding Row names
    modellist = pd.read_csv("m6Amodellist.csv",header=0,sep=",")
    global a
    for i in a:
        print(i)
        module = modellist[modellist['module'] == 'ME'+ str(i)]
        module.reset_index(inplace=True,drop=True)
        MEn = pd.read_csv('./Test_output/ME'+str(i)+'_pre.csv',header=0,sep=",")
       # B = DataFrame(np.append(module,MEn,axis=1))
        B = module.join(MEn)
        B.to_csv('./Test_output/ME'+str(i)+'_pre.csv', index = False, header = True, mode='w')

        
if __name__ == '__main__':
    runmodels()
    addC()
    addR()
    print('Scm6A done')
