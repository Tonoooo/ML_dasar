import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

kangker = datasets.load_breast_cancer()
x = kangker.data
y = kangker.target
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2) #0.2 = 20%
classes = ['malignant','benign']

"""
## -------------------- Menerapkan SVM (ini versi sederhana) ---------------------------------
# Menerapkan SVM sebenarnya cukup mudah. Kita cukup membuat model baru dan memanggil .fit() pada data pelatihan kita.
clff = svm.SVC() # ini menerapkan svm nya.  SVC = suport vektor clasifikasion.
clff.fit(x_train,y_train) # ini membuat modelnya

## menilai data kami,
# kami akan menggunakan alat yang berguna dari modul sklearn.
y_predd = clf.predict(x_test)  # Prediksi nilai untuk data pengujian kami
accc = metrics.accuracy_score(y_test, y_predd) # Uji mereka terhadap nilai-nilai yang benar kami
print(accc)
"""

##--------------------- Menerapkan SVM (ini versi yang lebih MANTAP) ----------------------
# bedanya dengan model yang diatas, disini ditambah kan parameter agar lebih akurat

## menambahkan parameter kernel (menjadikan 3d)
# tambahkan kernel agar jadi 3d
# "linear" untuk agar berpangkat 2 (x1^2 + x2^2 = x3)
# C adalah margin lunak/soft (poin/titik yang diizinkan berada dalam margin) # ini bebas, boleh pake boleh tidak, tapi agar prediksinya lebih akurat kita gunakan parameter c ini
clf = svm.SVC(kernel="linear", C=1)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
acc = metrics.accuracy_score(y_test, y_pred)
print(acc)


# kernel memiliki opsi yaitu:
# - linear # ini  akan berpangkat 2 (x1^2 + x2^2 = x3)
# - poly   # ini  akan berpangkat 6/7 (x1^7 + x2^7 = x3)
# - rbf
# - sigmoid
# - precomputed