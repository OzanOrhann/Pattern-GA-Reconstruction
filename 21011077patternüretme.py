import cv2
import numpy as np
import os
from sklearn.cluster import KMeans

# Görseli belirtilen boyuta indirger ve ikili (binary) formata çevirir.
def preprocess_image_to_binary(image_path, size=(24, 24), threshold=127):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # Görüntü gri tona çevrilerek okunur
    if img is None:
        raise ValueError(f"Resim yüklenemedi: {image_path}")
    resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA) # Görüntü 24x24 boyutuna getirilir
    _, binary = cv2.threshold(resized, threshold, 1, cv2.THRESH_BINARY) # Binary hale getirilir (0 veya 1)
    return binary.astype(np.uint8)
# Binary görseli dosyaya 0-255 formatında kaydeder (0 siyah, 255 beyaz)
def save_binary_image(binary_img, output_path):
    cv2.imwrite(output_path, binary_img * 255)
# Verilen klasördeki tüm 24x24 görsellerden 3x3 blokları çıkarır (flatten edilmiş haliyle)
def extract_3x3_blocks_from_all_images(folder_path):
    all_blocks = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
            if binary.shape[0] % 3 != 0 or binary.shape[1] % 3 != 0:
                continue # Boyutu 3x3'e tam bölünmeyenler atlanır
            for i in range(0, binary.shape[0], 3):
                for j in range(0, binary.shape[1], 3):
                    block = binary[i:i+3, j:j+3].flatten() # 3x3 blok vektöre çevrilir
                    all_blocks.append(block)
    return np.array(all_blocks)
# 3x3 blokları k-means ile kümele ve her kümenin merkezini pattern olarak kaydet
def create_and_save_clustered_3x3_patterns(input_folder_24, output_folder_3, n_patterns=7):
    blocks = extract_3x3_blocks_from_all_images(input_folder_24)  # Tüm blokları çıkar
    kmeans = KMeans(n_clusters=n_patterns, random_state=42, n_init=10).fit(blocks)  # Kümeleme yapılır
    centroids = kmeans.cluster_centers_.reshape((n_patterns, 3, 3)) # Küme merkezleri yeniden 3x3 forma getirilir
    binary_centroids = (centroids > 0.5).astype(np.uint8)  # Merkezler threshold ile binary formata çevrilir
    for i, pattern in enumerate(binary_centroids, start=1):
        path = os.path.join(output_folder_3, f"pattern{i}_3x3.png")
        save_binary_image(pattern, path)  # Her pattern dosyaya kaydedilir
        print(f"Pattern {i} (k-means) kaydedildi: {path}")
# Belirli bir ikon seti için tüm işlemleri gerçekleştiren fonksiyon
def process_ikon_set(set_name):
    input_folder = os.path.join("resimler", set_name)
    output_folder_24 = os.path.join("resimler", f"{set_name}_24x24")
    output_folder_3 = os.path.join("resimler", f"{set_name}_3x3")
    os.makedirs(output_folder_24, exist_ok=True)
    os.makedirs(output_folder_3, exist_ok=True)

    # Tüm görüntüleri 24x24 boyutuna indirger ve kaydeder
    all_filenames = sorted([f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))])
    for filename in all_filenames:
        input_path = os.path.join(input_folder, filename)
        name, ext = os.path.splitext(filename)
        try:
            binary_24 = preprocess_image_to_binary(input_path, size=(24, 24))
            save_binary_image(binary_24, os.path.join(output_folder_24, f"{name}_24x24.png"))
            print(f"{filename} - 24x24 hedef resim olarak kaydedildi.")
        except Exception as e:
            print(f"{filename} işlenemedi: {e}")

    # Oluşturulan 24x24 görsellerden pattern'leri üret
    create_and_save_clustered_3x3_patterns(output_folder_24, output_folder_3)


if __name__ == "__main__":
    process_ikon_set('ikon_set1')
    process_ikon_set('ikon_set2')
    process_ikon_set('ikon_set3')
