# Importowanie niezbędnych bibliotek
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Funkcja do najbliższego sąsiedztwa w demosaicingu
def closest_neigh_demosaicing(img, mask):
    height, width, _ = img.shape
    demosaiced_image = np.zeros_like(img, dtype=np.uint8)

    # Iteracja po pikselach obrazu
    for w in range(height):
        for k in range(width):
            for kl in range(3):  # Iteracja po kanałach kolorów (R, G, B)
                if mask[kl, w % 2, k % 2] == 1:
                    demosaiced_image[w, k, kl] = img[w, k, kl]
                else:
                    # Obliczanie najbliższego sąsiada
                    closest_w = w + 1 if w % 2 == 0 else w - 1
                    closest_k = k + 1 if k % 2 == 0 else k - 1
                    # Sprawdzenie, czy sąsiedni piksel mieści się w granicach obrazu
                    if 0 <= closest_w < height and 0 <= closest_k < width:
                        demosaiced_image[w, k, kl] = img[closest_w, closest_k, kl]
                    else:
                        # Obsługa sytuacji, gdy indeksy są poza zakresem
                        demosaiced_image[w, k, 0] = demosaiced_image[w, k, 1] = demosaiced_image[w, k, 2] = 255

    return demosaiced_image


# Funkcja główna
def main():
    # Wczytanie obrazu i konwersja do przestrzeni kolorów RGB
    img = cv2.cvtColor(cv2.imread('4demosaicking (2).bmp'), cv2.COLOR_BGR2RGB)
    original_img = img.copy()
    height, width, _ = img.shape

    # Bayer - aplikacja maski Bayera do obrazu
    bayer_mask = np.array([[[0, 1], [0, 0]],  # R
                           [[1, 0], [0, 1]],  # G
                           [[0, 0], [1, 0]]], np.uint8)  # B

    for w in range(height):
        for k in range(width):
            # Zastosowanie reguł Bayera do zerowania pewnych kolorów w obrazie
            if (k % 2 == 0 and w % 2 == 0) or (k % 2 == 1 and w % 2 == 1):
                img[w, k, 0] = img[w, k, 2] = 0
            elif k % 2 == 0 and w % 2 == 1:
                img[w, k, 1] = img[w, k, 2] = 0
            else:
                img[w, k, 0] = img[w, k, 1] = 0

    # XTrans - aplikacja maski XTrans do obrazu
    xtrans_mask = np.array([[[0, 0, 0, 0, 1, 0],
                             [1, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1],
                             [0, 1, 0, 0, 0, 0]],  # R
                            [[1, 0, 1, 1, 0, 1],
                             [0, 1, 0, 0, 1, 0],
                             [1, 0, 1, 1, 0, 1],
                             [1, 0, 1, 1, 0, 1],
                             [0, 1, 0, 0, 1, 0],
                             [1, 0, 1, 1, 0, 1]],  # G
                            [[0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 1, 0, 1],
                             [0, 1, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0],
                             [1, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0]]], np.uint8)  # B

    ximg = np.zeros_like(original_img)

    # Mnożenie obrazu przez maskę XTrans
    for w in range(height):
        for k in range(width):
            for kl in range(3):  # Iteracja po kanałach kolorów (R, G, B)
                ximg[w, k, kl] = xtrans_mask[kl, w % 6, k % 6] * original_img[w, k, kl]

    # Demosaicing obrazów z użyciem funkcji
    demosaiced_img_bayer = closest_neigh_demosaicing(original_img, bayer_mask)
    demosaiced_img_xtrans = closest_neigh_demosaicing(original_img, xtrans_mask)

    # Porównanie oryginalnego obrazu z demosaiced
    bayer_por = original_img - demosaiced_img_bayer

    # Wyświetlanie obrazów
    plt.subplot(1, 6, 1)
    plt.imshow(original_img)
    plt.title('Oryginal')
    plt.axis('off')

    plt.subplot(1, 6, 2)
    plt.imshow(img)
    plt.title('Mozaika-Bayer')
    plt.axis('off')

    plt.subplot(1, 6, 3)
    plt.imshow(ximg)
    plt.title('Mozaika-XTrans')
    plt.axis('off')

    plt.subplot(1, 6, 4)
    plt.imshow(demosaiced_img_bayer)
    plt.title('Demozaika-Bayer')
    plt.axis('off')

    plt.subplot(1, 6, 5)
    plt.imshow(demosaiced_img_xtrans)
    plt.title('Demozaika-XTrans')
    plt.axis('off')

    plt.subplot(1, 6, 6)
    plt.imshow(bayer_por)
    plt.title('Porownanie Bayer')
    plt.axis('off')

    plt.show()


if __name__ == "__main__":
    main()