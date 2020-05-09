import pandas as pd
from sklearn.tree import DecisionTreeRegressor


csvfile = 'melb_data.csv'
df  = pd.read_csv(csvfile)
print(df.columns)
df = df.dropna(axis=0)

keyfeatures = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = df[keyfeatures]
y = df.Price

mel = DecisionTreeRegressor(random_state=1)
print(mel.fit(X, y))

print(X.head())
print(mel.predict(X.head()))

