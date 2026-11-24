import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from xgboost import plot_importance
import os 
import pickle

data="student-scores.csv"
df=pd.read_csv(data)

print(df.columns)

df.rename(columns={'first_name':'fn','last_name':'ln','part_time_job':'ptj','absence_days':'abd','extracurricular_activities':'exa','weekly_self_study_hours':'sth',
          'career_aspiration':'ca','math_score':'ms','history_score':'hs','physics_score':'ps','chemistry_score':'cs','biology_score':'bs',
          'english_score':'es','geography_score':'gs'},inplace=True)

df.dropna(inplace=True)
print(df.columns)


df['grade']=df[['ms', 'hs', 'ps', 'cs', 'bs', 'es', 'gs']].mean(axis=1)
# print(df['grade'].head())
# print(df.head())

lb=LabelEncoder()
df['gender']=lb.fit_transform(df['gender'])
df['ptj']=lb.fit_transform(df['ptj'])
df['exa']=lb.fit_transform(df['exa'])
df['ca']=lb.fit_transform(df['ca'])
df['abd']=pd.to_numeric(df['abd'], errors='coerce')
df['sth']=pd.to_numeric(df['sth'], errors='coerce')

x=df[['gender', 'ptj', 'abd', 'exa', 'sth', 'ca']]
# print(x.head())
y=df['grade']


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=32)

model=XGBRegressor(
        n_estimators=150,
        learning_rate=0.03,
        max_depth=5,
        objective="reg:squarederror",
        random_state=42
    )
model.fit(x_train,y_train)
    
pred=model.predict(x_test)

if not os.path.exists('xmodel'):
    os.makedirs('Xmodel')
    
with open('xmodel/perfomate.pkl', 'wb') as f:
    pickle.dump(model, f)

print(mean_absolute_error(y_test,pred))
print(mean_squared_error(y_test, pred))
print(r2_score(y_test, pred))


plot_importance(model)
plt.show()