import pandas as pd
import numpy as np  # hal hal yang dilakukan disini(trutama sklearn) membutuhkan array bukan list
import sklearn
from sklearn import linear_model

### Linear regression

data = pd.read_csv("student-mat.csv",sep=";") # Karena data kita dipisahkan oleh titik koma, kita perlu melakukan sep=";"
#print( data.head()) # melihat dataframe nya (lima baris pertama saja)

## Memangkas Data
# Karena kita memiliki begitu banyak atribut(atribut bisa dibilang kolom) dan tidak semuanya relevan, kita perlu memilih atribut/(kolom) yang ingin kita gunakan.
# kolom kolom seperti G1,G2,G3,studytime,  dikenal sebagai atribut
data = data[["G1","G2","G3","studytime","failures","absences"]] # ini nama kolomnya.
#print(data.head())

## Memisahkan Data
# kita perlu memisahkannya menjadi 4 array. Namun, sebelum kita dapat melakukannya, kita perlu mendefinisikan
# atribut apa yang ingin kita prediksi. Atribut ini dikenal sebagai label . Atribut lain yang akan menentukan
# label kita dikenal sebagai features . Setelah kita selesai melakukannya, kita akan menggunakan numpy untuk
# membuat dua array. Satu yang berisi semua fitur kami dan satu yang berisi label kami.
# label adalah apa yang anda ingin untuk mendapatkan apa yang anda cari dengan benar dan anda bisa memiliki banyak label.
# fitur adalah  Atribut lain yang akan menentukan label
prediksi = "G3"  # variabel predict ini berisi atribut/kolom G3, G3 adalah nilai akhir
X  =  np.array(data.drop([prediksi],axis=1)) # drop() untuk menghapus kolom/baris, 1 adlah axisnya,axis defaultnya 0(0/1 itu index),0 berarti baris kalo 1 itu kolom,karna 'G3' itu kolom maka axis nya 1. ini Fitur
y  =  np.array(data[prediksi])  # Label

# Setelah ini kita perlu membagi data kita menjadi data pengujian(test) dan pelatihan(train). Kami akan menggunakan 90%
# dari data kami untuk melatih(train) dan 10% lainnya untuk menguji(test). Alasan kami melakukan ini adalah agar kami tidak
# menguji model kami pada data yang telah dilihatnya.

# kita sudah memiliki variable fitur yaitu x dan variabel label yaitu y. maka kita menjadikan 4 variabel
# kan ada 4 variabel nah itu sudah ada urutannya (ingat harus sesuai urutan, tapi nama variabelnya bebas):
# varaibel pertama: untuk x train, variabel kedua: untuk x test, variabel ketiga: untuk y train, variabel keempat: untuk y test. (nama variabelnya bebas, yang terpenting urutannya)
x_train , x_test, y_train,  y_test  =  sklearn.model_selection.train_test_split(X,y, test_size = 0.1) # ini akan secara random membaginya, hanya 10% untuk menguji dan sisnya 90% untuk melatih, ini urutan code nya harus begini kecual nama varabel&nilai test_size bebas
# 0.1 adalah 10% untuk menguji(test)
# x_test ,  y_test digunakan untuk menguji keakuratan

## Menerapkan Algoritma(membuat modelnya/ml)
#Kita akan mulai dengan mendefinisikan model yang akan kita gunakan
linear = linear_model.LinearRegression() # jadi ini kita akan menggunakan linear regresi

#Selanjutnya kita akan melatih dan menilai model kita menggunakan array yang kita bua
linear.fit(x_train,y_train) # menyesuaikan data untuk MENEMUKAN garis yang paling cocok.
acc = linear.score(x_test,y_test) #disini kita menilai model kita. jadidia akan menghasilkan seberapa akurat modelnya. (contoh hasilnya: 0.86548756 artinya model 86% akurat dengan data aslinya)
print("keakuratannya: ",acc) #Untuk melihat seberapa baik kinerja algoritme pada data pengujian Untuk set data khusus ini,
# kita telah selesai membuat model/ml nya


## Melihat Konstanta, kita ingin melihat konstanta yang digunakan untuk menghasilkan garis
# koefisien adalah bilangan yg menyertai variabel contoh : 2x berarti koefisiennya x adalah 2, 3y berarti koefisiennya y adalah 3
print("koefisien: \n", linear.coef_)   # Ini adalah setiap nilai kemiringan dan ini mengahasilkan koefisien dari kelima variabrl
# intersep adlah Perpotongan sumbu Y
print('intersep: \n', linear.intercept_) # Ini adalah intersep


## Memprediksi setiap siswa
# Melihat nilai skor itu keren, tetapi saya ingin melihat seberapa baik algoritme kami bekerja pada setiap siswa.
# jadi disini kita akan melihat dan membandingkan hasil prediksi dengan data asli
memprediksi = linear.predict(x_test) # Dapatkan daftar semua prediksi
for i in range(len(memprediksi)): # len() digunakan untuk mengetahui panjang (jumlah item atau anggota) dari objek seperti sequence (string, list, tuple, range) dan collection (dictionary, set, dan frozenset).
    print(memprediksi[i],x_test[i],y_test[i])
# setelah dirun kita akan membandingkan hasil prediksi dengan data asli:
# contoh:  11.0373..  [13 11 3 1 40] 11
# penjelasan :
# 11.037..       --> ini hasil prediksinya nilai
# [13 11 3 1 40] --> ini data asli atribut
# 11             --> data asli nilai
# jadi 11.037..  ini hasil nilai prediksinya, lumayan sama dengan data nilai aslinya 11



## jadi:
# Regresi Linier adalah proses menemukan garis yang paling sesuai dengan titik data yang tersedia pada plot,
# sehingga kita dapat menggunakannya untuk memprediksi nilai output untuk input yang tidak ada dalam kumpulan
# data yang kita miliki, dengan keyakinan bahwa output tersebut akan jatuh pada garis