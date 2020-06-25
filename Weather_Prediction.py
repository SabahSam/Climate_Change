import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


#In Zweite Schritt werden die orginale Daten in das Programm importiert. Dies sind Orginale Daten und enthalten keine Manipulation oder Cleaning.

# Data for the CO2 mesurment.
co2 = pd.read_csv('co2-mm-mlo_csv.csv', delimiter=',')
# File tor the Average Temp mesurment
file = pd.read_csv('GlobalLandTemperaturesByCountry.csv', delimiter=',')

#Beide Dataeinen wurden vom Kaggle heruntergeladen.
# Da die Datein als CSV gespeichert sind, enthalten die eine komma seperatur.
# Dies wird mit dem Befehl delimiter=',' sortiert. Pandas enthält die Funktion read_csv welches zunächst die Daten auf co2 bzw. auf file geschrieben hat

co2_filter = co2[['Decimal Date','Trend']]


dic = {
    "Datum":[], "Trend":[], "Average":[]
}


co2.describe()


viz = co2['Trend']
viz.hist()
plt.show()


for i in range(1960,2014):
    co2_filter = co2[['Decimal Date', 'Trend']]
    mask = (co2_filter['Decimal Date'].astype(int) == i)
    temp = co2_filter.loc[mask]

    vc = file[['dt', 'AverageTemperature', 'Country']]
    mask1 = (vc['dt'].str.contains(str(i))) & (vc['Country'].str.contains('Germany'))
    vc = vc.loc[mask1]
    # Add data from Bothe Files to the Dictionary
    dic["Datum"].append(i)
    dic["Trend"].append(temp['Trend'].mean())
    dic["Average"].append(vc['AverageTemperature'].max())


viz_2 = vc['AverageTemperature']
viz_2.hist()
plt.show()



X = dic["Datum"]
Y = dic["Trend"]
Z = dic["Average"]

print("Length of X: ",len(X))
print("Length of Y: ",len(Y))
print("Length of Z: ",len(Z))



X_train,Y_train,X_test,Y_test = train_test_split(X,Y,train_size=0.2,random_state=0)
lin_reg = LinearRegression()



X = np.array(X).reshape(-1,1)
Y = np.array(Y).reshape(-1,1)
lin_reg.fit(X,Y)


plt.scatter(X,Y,color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Linear Regression')
plt.xlabel('Year')
plt.ylabel('Trend')
plt.show()


poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly,Y)

#Passing the Linear regression to the Polynomail


plt.scatter(X,Y,color='red')
plt.plot(X,pol_reg.predict(poly_reg.fit_transform(X)),color='green')
plt.title('Linear Poly Reg')
plt.xlabel('Date')
plt.ylabel('Trend')
plt.show()

#Daten in Dictionary schreiben. Index ist hier einzuhalten.
for s in range(2014,2050):

    dic["Trend"].append(float(str(pol_reg.predict(poly_reg.fit_transform([[s]])))[1:-1][1:-1]))



X_train,Z_train,X_test,Z_test = train_test_split(X,Z,train_size=0.2,random_state=0)
lin_reg_z = LinearRegression()
Z = np.array(Z).reshape(-1,1)
lin_reg_z.fit(X,Z)


poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly,Z)

#Die deutsche durchschnittliche Max Temperatur zeigt fast eine Lineare verhalten.


plt.scatter(X,Z,color='red')
plt.plot(X,lin_reg_z.predict(X),color='blue')
plt.title('Linear Regression')
plt.xlabel("Datum")
plt.ylabel("Average Temperature")
plt.show()

for f in range(2014,2050):
    dic["Datum"].append(f)
    dic["Average"].append(float(str(pol_reg.predict(poly_reg.fit_transform([[f]])))[1:-1][1:-1]))


#Hier fehlt das Random Error! T_T

plt.scatter(dic["Datum"],dic["Average"],color='red')
plt.plot(X,pol_reg.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('Polynomial Regression')
plt.xlabel("Datum")
plt.ylabel("Average Temperature")
plt.show()



F1 = dic["Datum"]
F2 = dic["Average"]
F1 = np.array(F1).reshape(-1,1)
F2 = np.array(F2).reshape(-1,1)
lin_reg_z.fit(F1,F2)
plt.scatter(F1,F2,color='red')
plt.plot(F1,pol_reg.predict(poly_reg.fit_transform(F1)),color='blue')
plt.title('Linear Regression')
plt.xlabel("Datum")
plt.ylabel("Average Temperature")
plt.show()

print("Generating Excel File...")
df = pd.DataFrame.from_dict(dic)
df.to_excel("Testing_Data.xlsx", index=False, startcol=0, startrow=0, header=True)
print("OK, File Saved")


def export_excel():
    print("Generating Excel File...")
    df = pd.DataFrame.from_dict(dic)
    df.to_excel("Testing_Data_Excel.xlsx", index=False, startcol=0, startrow=0, header=True)
    print("OK, File Saved")
    pass

def export_csv():
    print("Generating Excel File...")
    df = pd.DataFrame.from_dict(dic)
    df.to_csv("Testing_Data_csv.csv", index=False,header=True)
    print("OK, File Saved")
    pass


export_csv()
export_excel()