## KNN adalah singkatan dari K-Nearest Neighbors . KNN adalah algoritma pembelajaran mesin yang digunakan
# untuk mengklasifikasikan data. Alih-alih menghasilkan prediksi numerik seperti nilai siswa atau harga saham,
# ia mencoba mengklasifikasikan data ke dalam kategori tertentu.
## Singkatnya, K-Nearest Neighbors bekerja dengan melihat K titik terdekat dengan titik data
# yang diberikan (yang ingin kita klasifikasikan) dan memilih kelas yang paling banyak muncul untuk dijadikan nilai prediksi

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing #Iniakan digunakan untuk menormalkan data dan mengubah nilai non-numerik menjadi nilai numerik.

## Memuat Data
data = pd.read_csv("car.data")
print(data.head())  # Untuk memeriksa apakah data kita dimuat dengan benar

## merubah/Mengonversi Data
# Untuk melatih KNN kita harus mengonversi data string/apapun menjadi semacam angka. Beruntung bagi kami sklearn memiliki metode yang dapat melakukan ini untuk kami.
le = preprocessing.LabelEncoder() #membuat objek LabelEncoder dan kemudian menggunakannya untuk mengkodekan setiap kolom data kita menjadi integer.
buying = le.fit_transform(list(data["buying"])) # mengambil seluruh kolom buying dan merubah merjadi list dan kemudian merubah mereka menjadi nilai integer yang sesuai dan didalam array(intinya dari string menjadi array)
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
# Metod fit_transform() mengambil list (masing-masing kolom) dan akan mengembalikan kepada kami sebuah array yang berisi nilai-nilai baru kami.
# jadi itu merubah kolom yang isinya string menjadi integer dan merubah menjadi array

predict = "class"

## Mengelompokan dan merubah array tadi menjadi list
# kita perlu menggabungkan kembali data kita ke dalam daftar fitur dan daftar label. dan Kita dapat menggunakan fungsi zip() untuk mempermudah.
x = list(zip(buying, maint, door, persons, lug_boot, safety)) # zip() hanya membuat sekelompok objek tuple. dan ini fitur
y = list(cls) # label
# jadi pada dasarnya di x itu merubah kolomkolom mejadi satu list besar

### jadi tadi diawal datanya string dirubah menjadi array lalu dikelompokan dan kelompok tersebut jadi sebuah list. jadi akhirnya adalh sebuah list

# kita sudah memiliki variable fitur yaitu x dan variabel label yaitu y. maka kita menjadikan 4 variabel
# kan ada 4 variabel nah itu sudah ada urutannya (ingat harus sesuai urutan, tapi nama variabelnya bebas):
# varaibel pertama: untuk x train, variabel kedua: untuk x test, variabel ketiga: untuk y train, variabel keempat: untuk y test. (nama variabelnya bebas, yang terpenting urutannya)
x_train , x_test, y_train,  y_test  =  sklearn.model_selection.train_test_split(x, y, test_size = 0.1)
# 0.1 adalah 10% untuk menguji(test)
# x_test ,  y_test digunakan untuk menguji keakuratan
# x_train , x_test, y_train,  y_test = ini masih tetap sebuah list