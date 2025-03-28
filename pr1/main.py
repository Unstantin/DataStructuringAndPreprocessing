import math

from PIL import Image
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
from tqdm import tqdm


def extract_pixels(references, multidimensional_array):
    pixels = []
    for ref in references:
        ul = ref["ul"]
        dr = ref["dr"]
        patch = multidimensional_array[ul[0]:dr[0], ul[1]:dr[1], :]
        # Изменяем форму patch, чтобы он был двумерным (пиксели x каналы)
        pixels.append(patch.reshape(-1, patch.shape[2]))
    return np.vstack(pixels)


references = {
    "town": [
        {"ul": (270, 447), "dr": (290, 502)},
        {"ul": (174, 625), "dr": (191, 654)}
    ],
    "sea": [
        {"ul": (541, 144), "dr": (560, 186)},
        {"ul": (452, 639), "dr": (502, 679)}
    ],
    "ground": [
        {"ul": (69, 441), "dr": (90, 475)},
        {"ul": (142, 63), "dr": (196, 161)}
    ],
    "forest": [
        {"ul": (109, 533), "dr": (143, 577)}
    ]
}

numbers = ['03', '04', '06', '08', '09', '11', '12']
n = len(numbers)
images = []
for i in numbers:
    image = Image.open(f"EO_Browser_images/2025-03-03-00_00_2025-03-03-23_59_Sentinel-2_L2A_B{i}_(Raw).jpg")
    images.append(np.array(image))
image = Image.open("EO_Browser_images/2025-03-03-00_00_2025-03-03-23_59_Sentinel-2_L2A_True_color.jpg")
arr = np.array(image)

multidimensional_array = np.stack(images, axis=-1)

town_pixels = extract_pixels(references["town"], multidimensional_array)
sea_pixels = extract_pixels(references["sea"], multidimensional_array)
ground_pixels = extract_pixels(references["ground"], multidimensional_array)
forest_pixels = extract_pixels(references["forest"], multidimensional_array)
print(f"Town pixels: {len(town_pixels)}")
print(f"Sea pixels: {len(sea_pixels)}")
print(f"Ground pixels: {len(ground_pixels)}")
print(f"Forest pixels: {len(forest_pixels)}")

# Вычисление среднего и ковариационной матрицы
town_mean = np.mean(town_pixels, axis=0)
town_cov = np.cov(town_pixels, rowvar=False)

sea_mean = np.mean(sea_pixels, axis=0)
sea_cov = np.cov(sea_pixels, rowvar=False)

ground_mean = np.mean(ground_pixels, axis=0)
ground_cov = np.cov(ground_pixels, rowvar=False)

forest_mean = np.mean(forest_pixels, axis=0)
forest_cov = np.cov(forest_pixels, rowvar=False)

epsilon = 1e-2  # Маленькое значение для регуляризации
E = epsilon * np.eye(town_cov.shape[0])  # Единичная матрица, умноженная на epsilon

# Регуляризация ковариационных матриц
town_cov_reg = town_cov + E
sea_cov_reg = sea_cov + E
ground_cov_reg = ground_cov + E
forest_cov_reg = forest_cov + E

# Вычисление обратных матриц
town_inv_cov_reg = np.linalg.inv(town_cov_reg)
sea_inv_cov_reg = np.linalg.inv(sea_cov_reg)
ground_inv_cov_reg = np.linalg.inv(ground_cov_reg)
forest_inv_cov_reg = np.linalg.inv(forest_cov_reg)

def cal_dif(pixel, mean, inv_cov):
    diff = pixel - mean
    return np.sqrt(diff.T @ inv_cov @ diff)

# Классификация пикселей
classes_of_pixels = np.zeros((multidimensional_array.shape[0], multidimensional_array.shape[1]))
count1, count2, count3, count4 = 0, 0, 0, 0
for i in tqdm(range(multidimensional_array.shape[0])):
    for j in range(multidimensional_array.shape[1]):
        pixel = multidimensional_array[i, j, :]

        town_dif = cal_dif(pixel, town_mean, town_inv_cov_reg)
        sea_dif = cal_dif(pixel, sea_mean, sea_inv_cov_reg)
        ground_dif = cal_dif(pixel, ground_mean, ground_inv_cov_reg)
        forest_dif = cal_dif(pixel, forest_mean, forest_inv_cov_reg)

        min_dif = min(town_dif, sea_dif, ground_dif, forest_dif)
        if min_dif == town_dif:
            classes_of_pixels[i][j] = 1
            count1 += 1
        elif min_dif == sea_dif:
            classes_of_pixels[i][j] = 2
            count2 += 1
        elif min_dif == ground_dif:
            classes_of_pixels[i][j] = 3
            count3 += 1
        elif min_dif == forest_dif:
            classes_of_pixels[i][j] = 4
            count4 += 1



#print(classes_of_pixels)
print(count1, count2, count3, count4)
plt.imshow(classes_of_pixels, cmap='viridis')  # 'viridis' — одна из цветовых карт
plt.colorbar(label='Класс')  # Добавляем цветовую шкалу
plt.title('Результат классификации')  # Заголовок
plt.xlabel('Ось X')  # Подпись оси X
plt.ylabel('Ось Y')  # Подпись оси Y
plt.show()

plt.imshow(arr)
for cls, areas in references.items():
    for area in areas:
        ul, dr = area["ul"], area["dr"]
        rect = plt.Rectangle((ul[1], ul[0]), dr[1]-ul[1], dr[0]-ul[0],
                            fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
plt.show()