
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from pandas.plotting import scatter_matrix

names = ['id', 'TV_Budget', 'Radio_Budget', 'NP_Budget', 'Sales']
df = pd.read_csv("Advertising Budget and Sales.csv",names=names)
df = df.drop(columns=['id'])
df = df.drop(index = df.index[0])

df[['TV_Budget','Radio_Budget','NP_Budget','Sales']] = df[['TV_Budget','Radio_Budget','NP_Budget','Sales']].astype(object).astype(float)

# Drawing Histogram for the Given dataset
# df.hist(figsize=(10,10))

# # Pairplot
# sns.pairplot(df,
#              x_vars=['TV_Budget','Radio_Budget','NP_Budget'],
#              y_vars=['Sales'],
#              height= 5)
# plt.savefig('static/pairplot.png')


# Drawing Heatmap
# sns.heatmap(data=df.corr(),
#             annot = True,
#             linewidths=2,
#             linecolor='black',
#             fmt='.1g',
#             center=0.6)
# plt.savefig('static/heatmap.png')


# Applying Multi Linear Regression

y = df.Sales
x=df.drop(columns=['Sales'])
x=sm.add_constant(x)
x.head()

# Splitting of train and test data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2)

# Fitting the regression model

model = sm.OLS(y, x).fit()
print(model.summary())

# Getting the output
prediction = model.predict(x_test)
# print(prediction)

# Taking input from the user for the 
# new_input = [1]
# tv = int(input('enter the tv budget'))
# new_input.append(tv)
# radio = int(input('enter the radio budget'))
# new_input.append(radio)
# np = int(input('enter the newspaper budget'))
# new_input.append(np)

# Getting output for User defined data
# predictions = model.predict(new_input)
# print(predictions)

# Custom input
# x_test = [1, 100000, 20000, 40000]
# predictions = model.predict(x_test)
# print(predictions)

# Getting mean square error and mean absolute error
# from sklearn.metrics import mean_squared_error, mean_absolute_error
# print("MAE:", round(mean_absolute_error(y_test, prediction), 0))
# print("RMSE:", round(np.sqrt(mean_squared_error(y_test, prediction)), 0))

# Applying some graphs for visualizing the model and data  and output
# plt.plot(y_test, predictions, scalex= True, scaley= True)

# df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})

# df.plot(kind='line',figsize=(18,8))
# plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
# plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
# plt.show()

# df.cumsum()

# df.plot.barh()
# plt.figure(figsize = (20,30))

# data = pd.read_csv("Advertising Budget and Sales.csv")
# data = data.drop(columns = "Unnamed: 0")
# from pandas.plotting import radviz
# radviz(frame = data, class_column = "Sales ($)", color = "blue")

# data.plot(subplots = True, figsize=(15,10))

# (df.plot.box())

# df.plot.kde ()

# Saving the Model
joblib.dump(model, 'advertising_model.pkl')
