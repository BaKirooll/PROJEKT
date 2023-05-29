import numpy as np  # Import modułu numpy do manipulacji tablicami
from PIL import Image as im  # Import modułu PIL do obsługi obrazów

def convertToBW(imageIn):
    blackAndWhite = imageIn.convert('1')  # Konwersja obrazu na czarno-biały
    blackAndWhite = np.array(blackAndWhite) * 1  # Konwersja obrazu do tablicy numpy, zamiana wartości pikseli na 0 i 1
    return blackAndWhite


def toggleOnesAndZeros(blackAndWhite):
    return (blackAndWhite ^ 1)  # Zamiana wartości pikseli na 0 i 1 (przełączanie)

def modifyInputPixelsValues(blackAndWhite):
    in_blackAndWhite = blackAndWhite
    [numberRowPixels, numberColumnPixels] = blackAndWhite.shape

    for i in range(0, numberRowPixels):
        for j in range(0, numberColumnPixels):
            if in_blackAndWhite[i, j] > 0:
                in_blackAndWhite[i, j] = 3  # Modyfikacja wartości pikseli większych od zera na 3

    toggled = in_blackAndWhite
    return toggled


def crop(blackAndWhiteToggled):
    [numberOfRowPixels, numberOfColumnPixels] = blackAndWhiteToggled.shape

    verticalSumOfBlackPixels = np.sum(blackAndWhiteToggled, axis=0)  # Sumowanie wartości pikseli w pionowym kierunku
    leftDetected = False
    for i in range(0, numberOfColumnPixels):
        if verticalSumOfBlackPixels[i] > 0 and leftDetected == False:
            leftDetected = True
            left = i  # Wykrycie lewej krawędzi znaku
        elif verticalSumOfBlackPixels[i] > 0 and leftDetected == True:
            right = i  # Wykrycie prawej krawędzi znaku

    horizontalSumOfBlackPixels = np.sum(blackAndWhiteToggled, axis=1)  # Sumowanie wartości pikseli w poziomym kierunku
    topDetected = False
    for i in range(0, numberOfRowPixels):
        if horizontalSumOfBlackPixels[i] > 0 and topDetected == False:
            topDetected = True
            top = i  # Wykrycie górnej krawędzi znaku
        elif horizontalSumOfBlackPixels[i] > 0 and topDetected == True:
            bottom = i  # Wykrycie dolnej krawędzi znaku

    v_CroppedBlackAndWhite_array = blackAndWhiteToggled[:, (range(left, right + 1))]  # Przycięcie tablicy w pionie
    finalCroppedBlackAndWhite = v_CroppedBlackAndWhite_array[(range(top, bottom + 1)), :]  # Przycięcie tablicy w poziomie
    return finalCroppedBlackAndWhite


def normalize(character_in, width, height):
    character_in = im.fromarray(character_in)  # Konwersja tablicy numpy na obraz PIL
    normalized = character_in.resize((width, height), im.HAMMING)  # Normalizacja rozmiaru obrazu
    NormalizedArray = np.array(normalized)  # Konwersja obrazu do tablicy numpy
    return NormalizedArray
