import pandas as pd

data = {
    "Department" : ["IT","HR","IT","HR"],
    "Salary" : [50000,40000,60000,45000]
}
df = pd.DataFrame(data)
print(
    df.groupby("Department")["Salary"].median()
)

# df = pd.read_csv("Book1.csv")
# print(df.sort_values("Marks",ascending=False,inplace=True))
# print(df)

# df["Pass"] = df["Marks"] > 70
# df.drop("Pass",axis=1,inplace=True)
# print(df)



# df.loc[0,"Marks"] = 99
# print(df)



# df["Pass"] = df["Marks"] > 70
# print(df)

# print(df[df["Age"]>25])
# print(df)
# print(df.head())
# print(df.tail())
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.describe())
# print(df["Name"])
# print(df[["Age","Marks"]])
# print(df.loc[4])
# print(df.iloc[4]["Age"])
# df.index=[101,102,103,104,105]
# print(df)

# data = {
#     "Name" : ["Emp1","Emp2","Emp3"],
#     "Age" : [25,30,35],
#     "Salary" : [50000,60000,70000]
# }
# df = pd.DataFrame(data)
# print(df)

# data = [10,20,30,40]
# res = pd.Series(data)
# print(res)
# print(pd.__version__)