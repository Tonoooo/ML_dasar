#--------------------------- MASIH DI LINEAR REGRESSION DAN ini yang TERAKHIR

import pandas as pd
import numpy as np  # hal hal yang dilakukan disini(trutama sklearn) membutuhkan array bukan list
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import pickle  # untuk menyimpan model
from matplotlib import style  # untuk mengubah gaya/bentuk grib(yang seperti titik titik/bulet)

### menyimpan model (ketika kita latih modul, saat kita membuka kembali modul masa kita harus melatih lagi bagaimana
# jika modelnya besar akan memakan waktu, maka dari itu kita bisa simpan dan tidak perlu melatih lagi)

"""
# Untuk menyimpan model kita, kita akan menulis ke file baru menggunakan pickle.dump()
with open("studentmodel.pickle", "wb") as f: # "studentmodel.pickle" ini namanya, namanya boleh bebas tapi pake ".pickle"  ,"WB" = membukanya dalam mode WB, dan f = statemen ini sebagai f
    pickle.dump(linear, f) # disini kita simpan. linear adalah nama model yang kita buat di tutorial terakhir , dan f adalah (with open("studentmodel.pickle", "WB"))
# membaca pikle nya 
pickle_in = open("studentmodel.pickle", "rb") # "rb" = membukanya dalam mode rb
# kita memuat pikle ini kedalam model linear
linear = pickle.load(pickle_in)
"""




#----------- cara yang diatas oke tapi ada yang lebih baik lagi
# cara yang lebih baik Saving Models & Plotting Data
# kita akan menyimpan model sampai memdapatkan skor tertentu

data = pd.read_csv("student-mat.csv",sep=";")
data = data[["G1","G2","G3","studytime","failures","absences"]]

predict = "G3"
X  =  np.array(data.drop([predict],1))
y  =  np.array(data[predict])
x_train , x_test, y_train,  y_test  =  sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

## melatih model (mungkin bisa berkali kali dilatihnya) demi untuk skor terbaik
best = 0
for i in range(30): # model akan melakukan prediksi berkali kali sebanyak 30 kali
    x_train , x_test, y_train,  y_test  =  sklearn.model_selection.train_test_split(X, y, test_size = 0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train,y_train)
    acc = linear.score(x_test,y_test)
    print(acc)

    # Jika model saat ini memiliki skor yang lebih baik dari yang telah kita latih maka simpan
    if acc > best:
        best = acc
        # menyimpan modelnya
        with open("studentmodel.pickle", "wb") as f: # "studentmodel.pickle" ini namanya, namanya boleh bebas tapi pake ".pickle"  ,"wb" = membukanya dalam mode WB, dan f = statemen ini sebagai f
            pickle.dump(linear, f) # disini kita simpan. linear adalah nama model yang kita buat di tutorial terakhir , jadi si linear nya disimpan di f dan f sebagai (with open("studentmodel.pickle", "WB"))

## stelah dilatih lalu disimpan, dan ini sekarang memuat modelnya
pickle_in = open("studentmodel.pickle", "rb") # membaca pikle nya. "rb" = membukanya dalam mode rb
linear = pickle.load(pickle_in) # kita memuat pikle ini kedalam variabel/model linear

print(40*'-')
print('hasi skor prediksinya: \n',best)
print("Coefficient: \n", linear.coef_)
print('Intercept: \n', linear.intercept_)
print(40*'-')

predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x],x_test[x],y_test[x])

# Menggambar dan membuat plot model
p = 'absences' # jika ingin tahu grafik lainnya Ubah ini menjadi G1/G2/studytime/failures untuk melihat
style.use('ggplot') # gaya plotnya
plt.scatter(data[p], data["G3"]) # scatter((disini x nya), (disini y nya)). scatter= agar plotnya berbentuk titik titik
plt.legend(loc=4)
plt.xlabel(p)
plt.ylabel("Final Grade")
plt.show()