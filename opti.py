import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import nnls



df=pd.read_csv('data_velo.csv')

dt= 0.01

# filtrer à 3 chiffres après la virgule le dataframe
df['Vitesse [m/s]'] = df['Vitesse [m/s]']
df['I [A]'] = df['I [A]']
df['Couple pédale [Nm]'] = df['Couple pédale [Nm]']
df['Time [s]'] = df['Time [s]']


df['vitesse_décalée'] = df['Vitesse [m/s]'].shift(-1)





df = df.iloc[:-1]

df['B']=df['vitesse_décalée']-df['Vitesse [m/s]']

n=len(df)

# filtrer à 3 chiffres après la virgule


A= np.ones((n,4))
A[:,1]=-df['Vitesse [m/s]']
A[:,2]=df['I [A]']
A[:,3]=df['Couple pédale [Nm]']
B=np.array(df['B'])

print(A)
print(B)

A=dt*A


# On résout le problème en utilisant la méthode des moindres carrés

X = np.linalg.lstsq(A,B,rcond=None)[0]
print(X)
solution = nnls(A, B)[0]
print(solution)

df['Vitesse predite']=0
df['Vitesse predite'][0]=df['Vitesse [m/s]'][0]

for i in range(1,n):
    df['Vitesse predite'][i] = df['Vitesse predite'][i-1] + dt*(X[0] - X[1]*df['Vitesse predite'][i-1] + X[2]*df['I [A]'][i-1] + X[3]*df['Couple pédale [Nm]'][i-1])

plt.plot(df['Time [s]'],df['Vitesse predite'])
plt.plot(df['Time [s]'],df['Vitesse [m/s]'])
plt.show()


