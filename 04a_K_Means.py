"""
Algoritma pengelompokan K-Means adalah algoritma klasifikasi yang mengikuti
langkah-langkah yang diuraikan di bawah ini untuk mengelompokkan titik-titik data
bersama-sama. Ini mencoba untuk memisahkan setiap area dari ruang dimensi tinggi kita
menjadi bagian-bagian yang mewakili setiap kelas. Ketika kami menggunakannya untuk
memprediksi, itu hanya akan menemukan bagian mana dari poin kami dan menetapkannya
ke kelas itu.

Langkah 1 : Pilih titik K secara acak untuk menempatkan K centroids
Langkah 2 : Tetapkan semua titik data ke centroid berdasarkan jarak. Centroid terdekat ke
suatu titik adalah yang ditugaskan.
Langkah 3: Rata-rata semua titik milik masing-masing centroid untuk menemukan tengah cluster
tersebut (pusat massa). Tempatkan centroid yang sesuai ke dalam posisi itu.
Langkah 4 : Tetapkan kembali setiap titik sekali lagi ke centroid terdekat.
Langka
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

## Memuat Kumpulan Data
# Kita akan memuat kumpulan data dari modul sklearn dan menggunakan fungsi scale/skala untuk memperkecil data kita.
# Kami ingin mengonversi nilai besar yang terdapat sebagai fitur menjadi rentang antara -1 dan 1 untuk menyederhanakan perhitungan dan membuat pelatihan lebih mudah dan akurat.
#datanya:
digit = load_digits()
# firtunya:
data = scale(digit.data) #ini datanya di perkecil agar sesuai nilainya -1 sampai 1,alasanya karna data aslinya besar dan akan memakan waktu lama
# labelnya:
y = digit.target

#menentukan jumlah cluster dengan membuat variabel k
# mengatur jumlah cluster yang akan dicari untuk jumlah centroid yang akan dibuat
k = 10
sampel, fitur = data.shape # menentukan berapa banyak sampel dan fitur yang kami miliki dengan mendapatkan bentuk kumpulan data.

## menilai modelnya
# Untuk menilai model kami, kami akan menggunakan fungsi dari situs web sklearn. Ini menghitung banyak skor berbeda untuk berbagai bagian model kami
def bench_k_means(estimator, name, data):
    estimator.fit(data)
    print('%-9s\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, estimator.inertia_,
             metrics.homogeneity_score(y, estimator.labels_),
             metrics.completeness_score(y, estimator.labels_),
             metrics.v_measure_score(y, estimator.labels_),
             metrics.adjusted_rand_score(y, estimator.labels_),
             metrics.adjusted_mutual_info_score(y,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))

## melatih model
#Terakhir untuk melatih model kita akan membuat classifier K Means kemudian meneruskan classifier tersebut ke fungsi yang kita buat di atas untuk menilai dan melatihnya.
clf = KMeans(n_clusters=k, init="random", n_init=10) # melatih model
bench_k_means(clf,"namanya bebas",data)
# n_cluster = Jumlah cluster yang akan dibentuk serta jumlah centroid yang akan dihasilkan.
# init="random"  = mengubah lokasi centroid secara acak (jika 'k=means++' sama aja tapi lebih cepat)
# n_init = berapa kali kita akan menjalankan algoritma
# penjelasannya: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# atau https://youtu.be/zixd-si9Q-o


# setelah dirun:
# semakin besar nilai/angka nya maka semakin bagus (ya.. tidak semua begitu)
# penjelasannya: https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
print(50*"#")

####################################################################
# Contoh Visualisasi MatplotLib
# Untuk melihat representasi visual tentang cara kerja K Means, Anda dapat menyalin dan menjalankan kode ini dari komputer Anda. Ini dari dokumentasi SkLearn
# Visualize the results on PCA-reduced data
n_digits = len(np.unique(digit.target))
reduced_data = PCA(n_components=2).fit_transform(data)
kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
kmeans.fit(reduced_data)

# Ukuran langkah mesh. Penurunan untuk meningkatkan kualitas VQ.
h = .02     # titik di mesh [x_min, x_max]x[y_min, y_max].

# Plot batas keputusan. Untuk itu, kita akan menetapkan warna untuk setiap
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Dapatkan label untuk setiap titik di mesh. Gunakan model terlatih terakhir.
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

# Masukkan hasilnya ke dalam plot warna
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
# Plot centroid sebagai X putih
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()