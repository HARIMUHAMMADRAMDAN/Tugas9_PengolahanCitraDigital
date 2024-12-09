import imageio as img
import numpy as np
import matplotlib.pyplot as plt

# Membaca gambar dan mengonversi ke skala abu-abu (jika diperlukan)
image = img.imread('D:/PERKULIAHAN/SEMESTER 5/Pengolahan Citra Data/SESI 9/Tugas9/otter.jpg')
if len(image.shape) == 3:  # Jika gambar RGB, ubah ke grayscale
    image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

# Sobel kernel untuk deteksi tepi
sobelX = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

sobelY = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
])

# Menambahkan padding pada gambar
imgPad = np.pad(image, pad_width=1, mode='constant', constant_values=0)

# Matriks kosong untuk hasil gradient
Gx = np.zeros_like(image)
Gy = np.zeros_like(image)

# Perhitungan Gradien X dan Y
for y in range(1, imgPad.shape[0] - 1):
    for x in range(1, imgPad.shape[1] - 1):
        area = imgPad[y-1:y+2, x-1:x+2]
        Gx[y-1, x-1] = np.sum(area * sobelX)
        Gy[y-1, x-1] = np.sum(area * sobelY)

# Menghitung magnitudo gradien
G = np.hypot(Gx, Gy)  # sqrt(Gx^2 + Gy^2)
G = (G / G.max()) * 255
G = G.astype(np.uint8)

# Menampilkan hasil
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(Gx, cmap='gray')
plt.title("Gradient X")
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(Gy, cmap='gray')
plt.title("Gradient Y")
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(G, cmap='gray')
plt.title("Gradient Magnitude")
plt.axis('off')

plt.tight_layout()
plt.show()
