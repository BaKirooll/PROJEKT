def cropParagraphs(numberOfLines, topOfLines, bottomOfLines):
    """
    Funkcja cropParagraphs wykonuje wycięcie akapitów na podstawie podanych linii górnego i dolnego marginesu.

    Parametry:
    - numberOfLines: liczba linii w dokumencie
    - topOfLines: lista zawierająca indeksy linii górnego marginesu dla poszczególnych linii
    - bottomOfLines: lista zawierająca indeksy linii dolnego marginesu dla poszczególnych linii

    Zwraca:
    - locationOfNewLines: lista zawierająca indeksy linii, na których występuje nowy akapit

    """
    locationOfNewLines = []
    for i in range(1, numberOfLines):
        whiteSpaceDistance = topOfLines[i] - bottomOfLines[i - 1]
        if whiteSpaceDistance > 60:
            locationOfNewLines.append(i - 1)
    return locationOfNewLines


def cropLines(blackAndWhite):
    """
    Funkcja cropLines wykonuje wycięcie linii na podstawie obrazu w skali czerni i bieli.

    Parametry:
    - blackAndWhite: macierz reprezentująca obraz w skali czerni i bieli

    Zwraca:
    - croppedLinesList: lista wyciętych linii
    - numberOfLines: liczba wyciętych linii
    - topOfLines: lista zawierająca indeksy linii górnego marginesu dla poszczególnych linii
    - bottomOfLines: lista zawierająca indeksy linii dolnego marginesu dla poszczególnych linii

    """
    [numberRowPixels, numberColumnPixels] = blackAndWhite.shape
    h_firstBlackPixelDetected = False
    firstBlackPixelRow = 0
    lastBlackPixelRow = 0
    topOfLines = []
    bottomOfLines = []

    for i in range(0, numberRowPixels):
        sumOfAllPixelsInRow_i = sum(blackAndWhite[i, :])
        if sumOfAllPixelsInRow_i >= 1 and h_firstBlackPixelDetected == False:
            h_firstBlackPixelDetected = True
            firstBlackPixelRow = i
            lastBlackPixelRow = i

        elif sumOfAllPixelsInRow_i >= 1 and h_firstBlackPixelDetected == True:
            lastBlackPixelRow = i

        elif sumOfAllPixelsInRow_i < 1 and h_firstBlackPixelDetected == True:
            h_firstBlackPixelDetected = False
            topOfLines.append(firstBlackPixelRow)
            bottomOfLines.append(lastBlackPixelRow)

    numberOfLines = len(topOfLines)
    croppedLinesList = []
    for i in range(0, numberOfLines):
        croppedLine = blackAndWhite[(range(topOfLines[i], bottomOfLines[i])), :]
        croppedLinesList.append(croppedLine)

    return croppedLinesList, numberOfLines, topOfLines, bottomOfLines


def cropCharacters(croppedLine, numberOfLines):
    """
    Funkcja cropCharacters wykonuje wycięcie pojedynczych znaków na podstawie wyciętej linii.

    Parametry:
    - croppedLine: macierz reprezentująca wyciętą linię obrazu w skali czerni i bieli
    - numberOfLines: liczba wyciętych linii

    Zwraca:
    - croppedCharactersList: lista wyciętych znaków
    - numberOfCharacters: liczba wyciętych znaków
    - leftOfCharacters: lista zawierająca indeksy lewych marginesów dla poszczególnych znaków
    - rightOfCharacters: lista zawierająca indeksy prawych marginesów dla poszczególnych znaków

    """
    [numberOfLineRowPixels, numberOfLineColumnPixels] = croppedLine.shape
    leftDetected = False
    firstBlackPixelColumn = 0
    lastBlackPixelColumn = 0
    leftOfCharacters = []
    rightOfCharacters = []

    for i in range(0, numberOfLineColumnPixels):
        sumOfAllPixelsInColoumn_i = sum(croppedLine[:, i])
        if sumOfAllPixelsInColoumn_i >= 1 and leftDetected == False:
            leftDetected = True
            firstBlackPixelColumn = i
            lastBlackPixelColumn = i

        elif sumOfAllPixelsInColoumn_i >= 1 and leftDetected == True:
            lastBlackPixelColumn = i

        elif sumOfAllPixelsInColoumn_i < 1 and leftDetected == True:
            leftDetected = False
            leftOfCharacters.append(firstBlackPixelColumn)
            rightOfCharacters.append(lastBlackPixelColumn)

    numberOfCharacters = len(leftOfCharacters)
    croppedCharactersList = []
    for i in range(0, numberOfCharacters):
        croppedCharacter = croppedLine[:, (range(leftOfCharacters[i], rightOfCharacters[i]))]
        croppedCharactersList.append(croppedCharacter)

    return croppedCharactersList, numberOfCharacters, leftOfCharacters, rightOfCharacters


def cropWords(recognizedCharactersList, leftOfCharacters, rightOfCharacters):
    """
    Funkcja cropWords wykonuje wycięcie słów na podstawie wyciętych znaków.

    Parametry:
    - recognizedCharactersList: lista rozpoznanych znaków
    - leftOfCharacters: lista zawierająca indeksy lewych marginesów dla poszczególnych znaków
    - rightOfCharacters: lista zawierająca indeksy prawych marginesów dla poszczególnych znaków

    Zwraca:
    - words: lista wyciętych słów
    - numberOfWords: liczba wyciętych słów
z
    """
    numberOfCharacters = len(leftOfCharacters)
    locationOfSpaces = [0]
    words = []
    for i in range(1, numberOfCharacters):
        whiteSpaceBetweenCharacters = leftOfCharacters[i] - rightOfCharacters[i - 1]
        if whiteSpaceBetweenCharacters > 20:
            locationOfSpaces.append(i)
        if i == numberOfCharacters - 1:
            locationOfSpaces.append(numberOfCharacters)
    for i in range(0, len(locationOfSpaces) - 1):
        firstCharacterInWord = locationOfSpaces[i]
        lastCharacterInWord = locationOfSpaces[i + 1]
        word = recognizedCharactersList[firstCharacterInWord:lastCharacterInWord]
        word = ''.join(word)
        words.append(word)
    numberOfWords = len(words)
    return words, numberOfWords
