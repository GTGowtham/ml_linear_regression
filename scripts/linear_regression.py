import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
from sklearn.linear_model import LinearRegression

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_path = os.path.join(project_root, "datasets", "Computers.csv")
output = os.path.join(project_root, "output")

# os.makedirs(output, exist_ok=True)

df = pd.read_csv(data_path)
print(df)

# get some initial insights from dataset especially that is checking mean
mean_value = df['Minutes'].mean()

plt.scatter(df['Units'], df['Minutes'], marker='*', color='green')
plt.axhline(y=mean_value, c="r")
plt.annotate("Mean repair time", xy=(7.5, mean_value+2))
plt.xlabel("Units")
plt.ylabel("Minutes")
plt.title("insights of mean_value")
mean_time_fig = os.path.join(output, "mean_time_insight.png")
plt.savefig(mean_time_fig)

# Time of randomly y-axis randomly taken x and c value for c models
model0 = df['Minutes'].mean()
model1 = 10+2*df['Units']
model2 = 8+12*df['Units']
model3 = 7+14*df['Units']
model4 = 4.5+15*df['Units']
model5 = 9+16*df['Units']

df['min_model_0'] = model0
df['min_model_1'] = model1
df['min_model_2'] = model2
df['min_model_3'] = model3
df['min_model_4'] = model4
df['min_model_5'] = model5

fig, ax = plt.subplots()
ax.scatter(x='Units', y='Minutes', data=df, label='random_model_time')

ax.plot(df['Units'], df['min_model_0'], color='yellow', label='model0')
ax.plot(df['Units'], df['min_model_1'], color='red', label='model1')
ax.plot(df['Units'], df['min_model_2'], color='blue', label='model2')
ax.plot(df['Units'], df['min_model_3'], color='green', label='model3')
ax.plot(df['Units'], df['min_model_4'], color='brown', label='model4')
ax.plot(df['Units'], df['min_model_5'], color='purple', label='model5')

ax.set_title('model_fit')
ax.set_xlabel('Units')
ax.set_ylabel('Minutes')
ax.legend()

model_fit_random = os.path.join(output, 'model_fit_random.png')
fig.savefig(model_fit_random, dpi=300, bbox_inches='tight')

# model observation
model0_obsmod_DataFrame = pd.DataFrame({'Units': df['Units'],
                                         'Minutes': df['Minutes'],
                                         'Predicted_time': df['min_model_1'],
                                         'Error': df['min_model_0'] - df['Minutes']})

model_0_obsmod = os.path.join(output, 'model_0_obs.csv')
model0_obsmod_DataFrame.to_csv(model_0_obsmod)

model1_obsmod_DataFrame = pd.DataFrame({'Units': df['Units'],
                                         'Minutes': df['Minutes'],
                                         'Predicted_time': df['min_model_1'],
                                         'Error': df['min_model_1'] - df['Minutes']})

model_1_obsmod = os.path.join(output, 'model_1_obsmod.csv')
model1_obsmod_DataFrame.to_csv(model_1_obsmod)

model2_obsmod_DataFrame = pd.DataFrame({'Units': df['Units'],
                                         'Minutes': df['Minutes'],
                                         'Predicted_time': df['min_model_2'],
                                         'Error': df['min_model_2'] - df['Minutes']})

model_2_obsmod = os.path.join(output, 'model_2_obs.csv')
model2_obsmod_DataFrame.to_csv(model_2_obsmod)

model3_obsmod_DataFrame = pd.DataFrame({'Units': df['Units'],
                                         'Minutes': df['Minutes'],
                                         'Predicted_time': df['min_model_3'],
                                         'Error': df['min_model_3'] - df['Minutes']})

model_3_obsmod = os.path.join(output, 'model_3_obs.csv')
model3_obsmod_DataFrame.to_csv(model_3_obsmod)

model4_obsmod_DataFrame = pd.DataFrame({'Units': df['Units'],
                                         'Minutes': df['Minutes'],
                                         'Predicted_time': df['min_model_4'],
                                         'Error': df['min_model_4'] - df['Minutes']})

model_4_obsmod = os.path.join(output, 'model_4_obs.csv')
model4_obsmod_DataFrame.to_csv(model_4_obsmod)

model5_obsmod_DataFrame = pd.DataFrame({'Units': df['Units'],
                                         'Minutes': df['Minutes'],
                                         'Predicted_time': df['min_model_5'],
                                         'Error': df['min_model_5'] - df['Minutes']})

model_5_obsmod = os.path.join(output, 'model_5_obs.csv')
model5_obsmod_DataFrame.to_csv(model_5_obsmod)


print(type(tuple(model0_obsmod_DataFrame.Error**2)))
st_model0=sum(model0_obsmod_DataFrame['Error']**2)
print(type(st_model0))
st_model1=sum(model1_obsmod_DataFrame['Error']**2)
st_model2=sum(model2_obsmod_DataFrame['Error']**2)
st_model3=sum(model3_obsmod_DataFrame['Error']**2)
st_model4=sum(model4_obsmod_DataFrame['Error']**2)
st_model5=sum(model5_obsmod_DataFrame['Error']**2)

# list of st error
st_final=pd.DataFrame({'model_0_st_error':[st_model0],
                       'model_1_st_error':[st_model1],
                       'model_2_st_error':[st_model2],
                       'model_3_st_error':[st_model3],
                       'model_4_st_error':[st_model4],
                       'model_5_st_error':[st_model5]})

st_final_out=os.path.join(output,'st_table_out_final.csv')
st_final.to_csv(st_final_out,index=False)

# Mathematically solved this using slope y=mx+c

x=df.Units
y=df.Minutes
x_y=x*y
sq_x=x**2
mean_x=x.mean()
mean_y=y.mean()
n=len(df)
denominator=sum(sq_x)-n*(mean_x)**2
numerator=sum(x_y)-n*mean_x*mean_y
m=numerator/denominator
c=mean_y-(m*mean_x)
df['manual_pred'] = m * x + c
print(f'co-efficient {m},intercept {c}')

best_fit_line = m*x+c
print(f"best_fit_line : {best_fit_line}")

best_fit_data=pd.DataFrame({'Units':df['Units'],
                            'Minutes':df['Minutes'],
                            'best_fit_data':best_fit_line})

best_fit_data_out=os.path.join(output,'best_fit_data.csv')
best_fit_data.to_csv(best_fit_data_out)

df['min_best_fit_model'] = best_fit_line

fig,ax=plt.subplots()
ax.scatter(x='Units',y='Minutes',data=df)
ax.plot(df['Units'], df['min_best_fit_model'], color='red', label='best_fit_model')
ax.set_title('best_fit_model')
ax.set_xlabel('Units')
ax.set_ylabel('Minutes')
ax.legend()

best_fit_model = os.path.join(output, 'best_fit_model.png')
fig.savefig(best_fit_model, dpi=300, bbox_inches='tight')

x=df[['Units']]
y=df['Minutes']

model=LinearRegression()
model.fit(x,y)
print(f"coefficients : {model.coef_}")
print(f"intercept : {model.intercept_}")



y_pred=model.predict(x)
print(f"y_pred : {y_pred}")

# model observation
sk_model_obsmod_DataFrame = pd.DataFrame({'Units': df['Units'],
                                         'Minutes': df['Minutes'],
                                         'Predicted_time': y_pred,
                                         'Error': y_pred - df['Minutes']})

model_obsmod = os.path.join(output, 'skmodel_obs.csv')
sk_model_obsmod_DataFrame.to_csv(model_obsmod)


fig,ax=plt.subplots()
ax.scatter(x='Units',y='Minutes',data=df)
ax.plot(df['Units'], y_pred, color='red', label='skmodel')
ax.set_title('skmodel')
ax.set_xlabel('Units')
ax.set_ylabel('Minutes')
ax.legend()

skmodel = os.path.join(output, 'skmodel.png')
fig.savefig(skmodel, dpi=300, bbox_inches='tight')

# --- MANUAL METRICS ---

# SST = Σ (y - mean_y)²
sst_manual = sum((y - mean_y) ** 2)

# SSE = Σ (y - ŷ_manual)²
sse_manual = sum((y - df['manual_pred']) ** 2)

# SSR = SST - SSE
ssr_manual = sst_manual - sse_manual

# R² = 1 - (SSE / SST)
r2_manual = 1 - (sse_manual / sst_manual)

manual_eval = pd.DataFrame({
    'SST_manual': [sst_manual],
    'SSE_manual': [sse_manual],
    'SSR_manual': [ssr_manual],
    'R2_manual': [r2_manual],
    'slope_m': [m],
    'intercept_c': [c]
})

manual_eval_out = os.path.join(output, "manual_evaluation.csv")
manual_eval.to_csv(manual_eval_out, index=False)

# ------------------------
# SKLEARN LINEAR REGRESSION
# ------------------------

X = df[['Units']]
Y = df['Minutes']

model = LinearRegression()
model.fit(X, Y)

df['sk_pred'] = model.predict(X)

# --- SKLEARN METRICS ---
mean_y = df['Minutes'].mean()

# SST same formula
sst_sk = sum((df['Minutes'] - mean_y) ** 2)

# SSE = Σ (y - ŷ)²
sse_sk = sum((df['Minutes'] - df['sk_pred']) ** 2)

# SSR = SST - SSE
ssr_sk = sst_sk - sse_sk

# R² from sklearn
r2_sk = model.score(X, Y)

sk_eval = pd.DataFrame({
    'SST_sklearn': [sst_sk],
    'SSE_sklearn': [sse_sk],
    'SSR_sklearn': [ssr_sk],
    'R2_sklearn': [r2_sk],
    'coef': [model.coef_[0]],
    'intercept': [model.intercept_]
})

sk_eval_out = os.path.join(output, "sklearn_evaluation.csv")
sk_eval.to_csv(sk_eval_out, index=False)

# ------------------------
# SAVE OBSERVATION TABLES
# ------------------------

manual_obs = pd.DataFrame({
    'Units': df['Units'],
    'Minutes': df['Minutes'],
    'manual_pred': df['manual_pred'],
    'Error': df['manual_pred'] - df['Minutes']
})

manual_obs.to_csv(os.path.join(output, "manual_model_observations.csv"), index=False)

sk_obs = pd.DataFrame({
    'Units': df['Units'],
    'Minutes': df['Minutes'],
    'sk_pred': df['sk_pred'],
    'Error': df['sk_pred'] - df['Minutes']
})
