import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
predict = "class"
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)
x_train , x_test, y_train,  y_test  =  sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

## membuat dan melatih model nya
model = KNeighborsClassifier(n_neighbors=9) # n_neighbors=5 maksudnya banyak tetangga(yang terdekat) yang harus dicari adalah 9 (harus ganjil jangan genap)

model.fit(x_train,y_train)
#lalu kita menilai model kita. jadidia akan menghasilkan seberapa akurat modelnya. (contoh hasilnya: 0.86548756 artinya model 86% akurat dengan data aslinya)
acc = model.score(x_test,y_test)
print("keakuratannya: ",acc)

## Menguji Modelnya
prediksi = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"] #Kita membuat list nama sehingga kita dapat mengubah prediksi bilangan bulat kita menjadi representasi stringnya
for i in range(len(prediksi)):
    print("Prediksinya: ", names[prediksi[i]],",Data: ", x_test[i], ", Data aslinya: ", names[y_test[i]])
    #Sekarang kita akan melihat tetangga dan jaraknya dari setiap titik dalam data pengujian kita
    n = model.kneighbors([x_test[i]], 9, True)
    print("N: ", n)

"""
penjelasan model.kneighbors([x_test[i]], 9, True) :
KNN memiliki metode unik yang memungkinkan kita melihat tetangga dari titik data tertentu. Kami dapat
menggunakan informasi ini untuk merencanakan data kami dan mendapatkan ide yang lebih baik tentang di mana model 
kami mungkin kurang akurat. Kita dapat menggunakan model.neighbors untuk melakukan ini.

Catatan: metode .neighbors mengambil input 2D, ini berarti jika kita ingin melewatkan satu titik data, kita perlu
mengelilinginya dengan [] agar bentuknya benar. karna data kita bukan 2d, agar 2d maka pake [] (contoh [x_test[i]])

Parameter: Parameter untuk .neighbors adalah sebagai berikut (jumlah parameternya 3 tidak boleh kurang): 
x/data(bentuknya harus 2D array),  n_neighbors/tetangga(int), return_distance/jarakAntarTitik(True or False).
return: Ini akan mengembalikan kepada kita sebuah array dengan indeks dalam data kita masing-masing tetangga.
Jika distance/jarakAntarTitik=True maka itu juga akan mengembalikan jarak ke setiap tetangga dari titik data kita.

setelah dirun ada 2 array :
N:  (array([[0.        , 1.        , 1.        , 1.        , 1.        ,1.        , 1.        , 1.        , 1.41421356]]), array([[ 171,  454, 1215, 1361, 1275,  936,  357,  945,  267]], dtype=int64))
array pertama = jaraknya
array kedua = index dari titik nya, dype=int64 = integer 64 bit
"""