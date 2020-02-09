# Importing the libraries
import numpy as np
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('house_data.csv')
dataset = dataset.drop(['id','date'], axis = 1)

with sns.plotting_context("notebook",font_scale=2.5):
    g = sns.pairplot(dataset[['sqft_lot','sqft_above','price','sqft_living','bedrooms']], 
                 hue='bedrooms', palette='tab20',size=6)
g.set(xticklabels=[])


X = dataset.iloc[:,1:].values
y = dataset.iloc[:,0].values
#splitting dataset into training and testing dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)


corr = dataset.corr()
import matplotlib.pyplot as plt
plt.figure(figsize=(25,25))
sns.heatmap(corr, annot=True, vmin = -1, vmax=1)



#Backward Elimination
import statsmodels.api as sm
X_opt=X[:,[0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
regressor_OLS=sm.OLS(y,X_opt).fit()
regressor_OLS.summary()





def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,17]]
X_Modeled = backwardElimination(X_opt, SL)


