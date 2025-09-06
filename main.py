import pandas as pd
from sklearn.tree import DecisionTreeRegressor

train_ds = pd.read_csv('new_train.csv')
test_ds = pd.read_csv('new_test_for_participants.csv')

x = train_ds.drop(columns=['ID','ViolentCrimesPerPop'])
y = train_ds['ViolentCrimesPerPop']
x_test = test_ds.drop(columns=['ID'])
test_id = test_ds['ID']

model = DecisionTreeRegressor(random_state=42)
model.fit(x,y)

predictions = model.predict(x_test)
result = pd.DataFrame({'ID':test_id,'ViolentCrimesPerPop':predictions})
result.to_csv('resultant_file.csv',index=False)
print('process completed resultant file has been completed.')