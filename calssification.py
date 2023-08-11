import pandas as pd

data = pd.read_csv("logr1.csv")
print(data.info())
print(data.head())

data["Gender"] = data["Gender"].map({"Male":0, "Female":1})
data["Admitted"] = data["Admitted"].map({"No":0, "Yes":1})


x = data[["SAT", "Gender"]]
y = data["Admitted"]

import statsmodels.api as sm
s = sm.add_constant(x)
res = sm.Logit(y, s).fit()
print(res.summary())



import numpy as np
np.set_printoptions(formatter={'float': lambda x:"{0:0.3f}".format(x)})

cls_result = res.predict()
print(cls_result)

print(cls_result > 0.5)

import matplotlib.pyplot as plt

plt.scatter(data["SAT"], data["Admitted"])
plt.scatter(data["SAT"], cls_result, c="red")
plt.show()

print(res.pred_table())

sat = int(input("Enter SAT score"))
gen = input("Male / Female ?")
gender = 0
if gen == "Female":
    gender=1

sample = pd.DataFrame({'const':1, 'SAT':[sat], 'Gender':[gender]})
sample = sample[["const", "SAT", "Gender"]]

admt = res.predict(sample)
print(admt)

if admt[0] > 0.65:
    print("Accepted")
else:
    print("Rejected")