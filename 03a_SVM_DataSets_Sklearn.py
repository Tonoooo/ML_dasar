# SVM adalah singkatan dari support vector machine. SVM biasanya digunakan untuk
# tugas klasifikasi yang serupa dengan yang kami lakukan dengan KNN.
## ------- disini kita hanya menggunakan datasets sklearn tidak membuat/melatih model

import sklearn
from sklearn import svm
from sklearn import datasets

# memuat datanya(di sklearn terdapat datasets kangker)
kangker = datasets.load_breast_cancer()

#  melihat daftar fitur dalam kumpulan data
print("Fitur: ", kangker.feature_names)
# melihat label fitur dalam kumpulan data
print("Label: ", kangker.target_names)

# memisahkan data
x = kangker.data  # Semua fitur nya dimasukkan ke x
y = kangker.target   # Semua label nya dimasukkan ke y

# kita sudah memiliki variable fitur yaitu x dan variabel label yaitu y. maka kita menjadikan 4 variabel
# kan ada 4 variabel nah itu sudah ada urutannya (ingat harus sesuai urutan, tapi nama variabelnya bebas):
# varaibel pertama: untuk x train, variabel kedua: untuk x test, variabel ketiga: untuk y train, variabel keempat: untuk y test. (nama variabelnya bebas, yang terpenting urutannya)
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2) #0.2 = 20%

print(x_train[:5], y_train[:5]) # :5 = dari index pertama sampai index 5/sebelum 5