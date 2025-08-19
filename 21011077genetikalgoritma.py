import cv2
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Genetik algoritma için hiperparametreleri döndüren fonksiyon
def get_hyperparams():
    return {
        'population_size': 50,    # Popülasyon büyüklüğü
        'num_generations': 200,   # Jenerasyon sayısı
        'num_parents': 10,        # Ebeveyn sayısı
        'mutation_rate': 0.1,     # Mutasyon oranı
        'elitism_count': 1        # Elit birey sayısı (doğrudan sonraki nesle aktarılacak)
    }

# Belirtilen klasörden pattern görsellerini okuyarak listeye dönüştürür
def create_patterns_from_files(pattern_folder):
    pattern_paths = [os.path.join(pattern_folder, f"pattern{i}_3x3.png") for i in range(1, 8)]
    patterns = []
    for path in pattern_paths:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Pattern yüklenemedi: {path}")
        _, binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY) # Binarize et
        patterns.append(binary.astype(np.uint8))
    return patterns


# Pattern indekslerine göre resmi yeniden oluşturur
def reconstruct_image(image, pattern_indices, patterns):
    reconstructed = np.zeros_like(image)
    for i in range(0, image.shape[0], 3):
        for j in range(0, image.shape[1], 3):
            pattern_index = pattern_indices[i // 3, j // 3]
            reconstructed[i:i + 3, j:j + 3] = patterns[pattern_index]
    return reconstructed

# Fitness değeri hesaplama (piksel benzerlik sayısı)
def fitness(original, reconstructed):
    return np.sum(original == reconstructed)

# Rastgele başlangıç popülasyonu oluşturur
def initialize_population(image_shape, population_size):
    return [np.random.randint(0, 7, size=(image_shape[0] // 3, image_shape[1] // 3))
            for _ in range(population_size)]

# Turnuva seçimi: Rastgele 5 birey seçilir, en iyisi ebeveyn olur
def select_parents(population, fitness_scores, num_parents):
    parents = []
    for _ in range(num_parents):
        tournament = random.sample(range(len(population)), 5)
        winner = tournament[np.argmax(fitness_scores[tournament])]
        parents.append(population[winner])
    return parents

# Çaprazlama işlemi: iki ebeveynden yeni birey üretimi
def crossover(parent1, parent2):
    rows, cols = parent1.shape
    crossover_row = random.randint(0, rows - 1)
    crossover_col = random.randint(0, cols - 1)
    child = np.copy(parent1)
    child[crossover_row:, crossover_col:] = parent2[crossover_row:, crossover_col:]
    return child

# Mutasyon işlemi: belirli oranda rastgele hücreyi değiştirir
def mutate(individual, mutation_rate):
    for i in range(individual.shape[0]):
        for j in range(individual.shape[1]):
            if random.random() < mutation_rate:
                individual[i, j] = random.randint(0, 6)
    return individual

# Pattern kullanım sayısını hesaplar
def get_pattern_usage_matrix(pattern_indices):
    usage_matrix = np.zeros((7,), dtype=int)
    for i in range(pattern_indices.shape[0]):
        for j in range(pattern_indices.shape[1]):
            usage_matrix[pattern_indices[i, j]] += 1
    return usage_matrix

# Orijinal ve yeniden oluşturulmuş görseli yan yana kaydeder
def save_comparison_image(original, reconstructed, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title("Orijinal")
    axes[0].axis('off')

    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Genetik algoritmanın tüm döngüsü
def genetic_algorithm(image, patterns, params, image_name=""):
    population = initialize_population(image.shape, params['population_size'])
    max_possible_fitness = image.size

    best_fitness_scores = []
    loss_values = []

    for generation in range(params['num_generations']):
        reconstructed_images = [reconstruct_image(image, individual, patterns) for individual in population]  # her bireyin skoru hesaplanır
        fitness_scores = np.array([fitness(image, recon) for recon in reconstructed_images])
        best_fitness = fitness_scores.max()
        best_fitness_scores.append(best_fitness)
        loss = max_possible_fitness - best_fitness
        loss_values.append(loss)

        sorted_indices = np.argsort(fitness_scores)[::-1]
        new_population = [population[i].copy() for i in sorted_indices[:params['elitism_count']]]

        parents = select_parents(population, fitness_scores, params['num_parents'])
        while len(new_population) < params['population_size']:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, params['mutation_rate'])
            new_population.append(child) #yeni bireyler oluşturulur

        population = new_population

    best_individual = population[np.argmax(fitness_scores)]
    best_img = reconstruct_image(image, best_individual, patterns)
    usage_matrix = get_pattern_usage_matrix(best_individual)

    return best_img, loss_values, usage_matrix, best_fitness_scores[-1], loss_values[-1], len(best_fitness_scores)

# Tek bir deney kümesini çalıştıran fonksiyon
def run_experiment(set_name, pop_size, mutation_rate, experiment_name):
    image_folder = f"resimler/{set_name}_24x24"
    pattern_folder = f"resimler/{set_name}_3x3"
    output_folder = f"deney_sonuclari/{experiment_name}"
    os.makedirs(output_folder, exist_ok=True)

    patterns = create_patterns_from_files(pattern_folder)
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))])

    summary_data = []
    comparison_data = []

    for image_file in image_files:
        img_path = os.path.join(image_folder, image_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        _, img_binary = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)

        if img_binary.shape[0] % 3 != 0 or img_binary.shape[1] % 3 != 0:
            continue

        params = get_hyperparams()
        params['population_size'] = pop_size
        params['mutation_rate'] = mutation_rate

        best_img, loss_values, usage_matrix, final_fitness, final_loss, last_gen = genetic_algorithm(
            img_binary, patterns, params, image_name=image_file)
        # Sonuçları kaydetme
        out_img_path = os.path.join(output_folder, image_file.replace(".png", "_reconstructed.png"))
        cv2.imwrite(out_img_path, best_img * 255)
        # Loss grafiği
        plt.figure()
        plt.plot(loss_values, label='Loss')
        plt.title(f"{image_file} - Loss")
        plt.xlabel("Generation")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(output_folder, image_file.replace(".png", "_loss.png")))
        plt.close()
        # Görsel karşılaştırma
        save_comparison_image(img_binary, best_img,
                              os.path.join(output_folder, image_file.replace(".png", "_compare.png")))

        for i, count in enumerate(usage_matrix):
            summary_data.append({
                'Image': image_file,
                'Pattern': f"Pattern {i + 1}",
                'Count': count,
                'Population Size': pop_size,
                'Mutation Rate': mutation_rate
            })

        comparison_data.append({
            'Image': image_file,
            'Population Size': pop_size,
            'Mutation Rate': mutation_rate,
            'Final Fitness': final_fitness,
            'Final Loss': final_loss,
            'Last Generation': last_gen
        })
    # CSV ve heatmap kayıt
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_folder, "pattern_usage_summary.csv"), index=False)

    df_compare = pd.DataFrame(comparison_data)
    df_compare.to_csv(os.path.join(output_folder, "experiment_summary.csv"), index=False)

    pivot = df_summary.pivot(index='Pattern', columns='Image', values='Count')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot.fillna(0), annot=True, cmap="Reds")
    plt.title(f"Pattern Kullanım Isı Haritası - {experiment_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "pattern_heatmap.png"))
    plt.close()

    print(f"Deney '{experiment_name}' tamamlandı.")

# Ana fonksiyon: Her set için 3 farklı hiperparametre kombinasyonu deneyi yapar
def main():
    for set_name in ["ikon_set1", "ikon_set2", "ikon_set3"]:
        run_experiment(set_name, 50, 0.1, f"deney1_{set_name}")
        run_experiment(set_name, 100, 0.1, f"deney2_{set_name}")
        run_experiment(set_name, 50, 0.3, f"deney3_{set_name}")

if __name__ == "__main__":
    main()