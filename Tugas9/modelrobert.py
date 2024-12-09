import imageio as img
import numpy as np
import matplotlib.pyplot as plt

def roberts_edge_detection(image):
    """
    Melakukan deteksi tepi menggunakan operator Roberts.
    """
    # Kernel Roberts
    robertsX = np.array([[1, 0], [0, -1]])  # Gradien X
    robertsY = np.array([[0, 1], [-1, 0]])  # Gradien Y

    # Konversi gambar ke tipe float untuk mencegah overflow
    image = image.astype(float)

    # Menambahkan padding pada gambar
    padded_image = np.pad(image, ((1, 0), (1, 0)), mode='constant', constant_values=0)

    # Matriks kosong untuk menyimpan hasil
    Gx = np.zeros_like(image)
    Gy = np.zeros_like(image)

    # Penerapan kernel Roberts
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i+2, j:j+2]  # Ambil area 2x2
            Gx[i, j] = np.sum(region * robertsX)
            Gy[i, j] = np.sum(region * robertsY)

    # Menghitung magnitudo gradien
    G = np.hypot(Gx, Gy)  # sqrt(Gx^2 + Gy^2)
    G = (G / G.max()) * 255  # Normalisasi ke [0, 255]
    return G.astype(np.uint8)

def main():
    # Membaca gambar input
    image_path = 'D:/PERKULIAHAN/SEMESTER 5/Pengolahan Citra Data/SESI 9/Tugas9/otter.jpg'  # Ubah dengan path gambar Anda
    image = img.imread(image_path, mode='L')  # Membaca gambar dalam skala abu-abu

    # Deteksi tepi menggunakan Roberts
    edges = roberts_edge_detection(image)

    # Menampilkan hasil
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Roberts Edge Detection")
    plt.imshow(edges, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
