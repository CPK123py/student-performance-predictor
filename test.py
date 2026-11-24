import pandas as pd
import pickle
import matplotlib.pyplot as plt
from xgboost import plot_importance
import matplotlib.pyplot as plt
from lime import lime_tabular
from sklearn.preprocessing import LabelEncoder


with open('Xmodel/perfomate.pkl', 'rb') as f:
    model=pickle.load(f)


data="student-scores.csv"
df=pd.read_csv(data)

print(df.columns)

df.rename(columns={'first_name':'fn','last_name':'ln','part_time_job':'ptj','absence_days':'abd','extracurricular_activities':'exa','weekly_self_study_hours':'sth',
          'career_aspiration':'ca','math_score':'ms','history_score':'hs','physics_score':'ps','chemistry_score':'cs','biology_score':'bs',
          'english_score':'es','geography_score':'gs'},inplace=True)

df.dropna(inplace=True)


df['grade']=df[['ms', 'hs', 'ps', 'cs', 'bs', 'es', 'gs']].mean(axis=1)

lb=LabelEncoder()
df['gender']=lb.fit_transform(df['gender'])
df['ptj']=lb.fit_transform(df['ptj'])
#df['abd']=lb.fit_transform(df['abd'])
df['exa']=lb.fit_transform(df['exa'])
#df['sth']=lb.fit_transform(df['sth'])
df['ca']=lb.fit_transform(df['ca'])

x=df[['gender', 'ptj', 'abd', 'exa', 'sth', 'ca']]
# print(x.head())
y=df['grade']

samp=pd.DataFrame([{
    'gender': 1,         
    'ptj': 0,           
    'abd': 0,            
    'exa': 0,             
    'sth': 0,         
    'ca': 0              
}])

# out=model.predict(['male','bartender',3,'football',40,'scientist'])
out=model.predict(samp)
print(out)
explainer = lime_tabular.LimeTabularExplainer(
    training_data=x.values,
    feature_names=x.columns.tolist(),
    mode='regression'
)

exp = explainer.explain_instance(
    data_row=samp.iloc[0].values,
    predict_fn=model.predict
)

exp.as_pyplot_figure()
plt.tight_layout()
plt.show()

