import liner  # Moduł zawierający funkcje do przycinania linii tekstu
import recognition as recog  # Moduł zawierający funkcje do rozpoznawania znaków
import normalize as normal  # Moduł zawierający funkcje do normalizacji obrazu
from PIL import Image as img  # Moduł PIL służący do obsługi obrazów

choose = input("Wybierz obraz od 1 do 4: ")
learn = float(input("Podaj współczynnik uczenia: "))
hidden = int(input("Podaj liczbę ukrytych neuronów: "))
target = float(input("Podaj docelowy błąd: "))

# Parametry sieci neuronowej
learning_rate = learn
target_error = target
number_hidden_neurons = hidden

# Parametry obrazu
width = 18
height = 16
momentum = 1

sample_number = 10
image_path = "PrzykladoweObrazy/" + choose + ".png"

print('Wczytywanie obrazu....')
image = img.open('%s' % image_path)
image_black_white = normal.convertToBW(image)  # Konwersja obrazu na czarno-biały
image_black_white = normal.toggleOnesAndZeros(image_black_white)  # Zamiana wartości pikseli na 0 i 1
image.show()
print('Obraz wczytany')

print('Trwa trening sieci neuronowej.....')
# Inicjalizacja wag i anomalii sieci neuronowej
weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output = recog.initializeWeights(
    width, height, number_hidden_neurons
)
# Trenowanie sieci neuronowej
weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output = recog.trainNet(
    weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output, height, width, sample_number,
    learning_rate, momentum, target_error
)
print('Sieć neuronowa wytrenowana')

# Przycinanie linii tekstu na obrazie
[croppedLinesList, numberOfLines, topOfLines, bottomOfLines] = liner.cropLines(image_black_white)
# Przycinanie akapitów na podstawie odstępów między liniami
locationOfNewLines = liner.cropParagraphs(numberOfLines, topOfLines, bottomOfLines)
numberOfNewLines = len(locationOfNewLines)
linesContents = []

for line in range(0, numberOfLines):
    # Przycinanie znaków na linii tekstu
    [croppedCharactersList, numberOfCharacters, leftOfCharacters, rightOfCharacters] = liner.cropCharacters(
        croppedLinesList[line], numberOfLines
    )
    recognizedCharacterlist = []

    for character in range(0, numberOfCharacters):
        # Przygotowanie przyciętego znaku do rozpoznania
        inputCroppedBW = normal.crop(croppedCharactersList[character])
        inputNormalized = normal.normalize(inputCroppedBW, width, height)

        # Rozpoznawanie znaku przy użyciu sieci neuronowej
        output = recog.recognizeCharacter(inputNormalized, weight_input_hidden, weight_hidden_output, anomaly_hidden,
                                          anomaly_output)
        recognizedCharacterlist.append(output)

    # Przycinanie wyrazów na podstawie odstępów między znakami
    [words, numberOfWords] = liner.cropWords(recognizedCharacterlist, leftOfCharacters, rightOfCharacters)
    linesContents.append(words)

print(linesContents)

# Zapis wynikowego tekstu do pliku
fileName = open('Wyjscie.txt', 'w')
newLinesIndex = 0
for line in range(0, len(linesContents)):
    if line > 0:
        fileName.write('\n')
    for word in range(0, len(linesContents[line])):
        fileName.write(linesContents[line][word])
        fileName.write(' ')

    if line == locationOfNewLines[newLinesIndex]:
        fileName.write('\n')
        if numberOfNewLines - 1 > newLinesIndex:
            newLinesIndex += 1
fileName.close()
