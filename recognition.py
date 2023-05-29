from PIL import Image as im  # Import modułu PIL do obsługi obrazów
import numpy as np  # Import modułu numpy do manipulacji tablicami
import matplotlib.pyplot as plt  # Import modułu matplotlib do tworzenia wykresów
import normalize as normal  # Import modułu normalize (prawdopodobnie lokalnego)

import liner  # Import modułu liner (prawdopodobnie lokalnego)

letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i',
           'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r',
           's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

def initializeWeights(width, height, number_hidden_neurons):
    weight_input_hidden = np.random.random(size=(number_hidden_neurons, height, width)) - 0.5  # Inicjalizacja macierzy wag dla warstwy wejściowej do ukrytej
    weight_hidden_output = np.random.random(size=(26, number_hidden_neurons)) - 0.5  # Inicjalizacja macierzy wag dla warstwy ukrytej do wyjściowej
    anomaly_hidden = np.random.random(number_hidden_neurons) - 0.5  # Inicjalizacja wektora anomałii dla warstwy ukrytej
    anomaly_output = np.random.random(26) - 0.5  # Inicjalizacja wektora anomałii dla warstwy wyjściowej

    return weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output

def logistic(summation):
    out = 1 / (1 + np.exp(-summation))  # Funkcja logistyczna (sigmoidalna)
    return out

def feedForward(normalized, weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output):
    n_h = 0
    [number_hidden_neurons, height, width] = weight_input_hidden.shape

    output_hidden_neurons = []
    net_input_hidden_neurons = []

    for hidden_neuron in range(0, number_hidden_neurons):
        for i in range(0, height):
            for j in range(0, width):
                WxP = weight_input_hidden[hidden_neuron, i, j] * normalized[i, j]  # Obliczanie iloczynu wag i piksela wejściowego
                n_h = n_h + WxP

        n_h = n_h + anomaly_hidden[hidden_neuron]
        output_hidden_neurons.append(logistic(n_h))  # Obliczanie wartości wyjściowej neuronu ukrytego
        net_input_hidden_neurons.append(n_h)
        n_h = 0

    out_hidden_x_weights = output_hidden_neurons * weight_hidden_output  # Mnożenie wartości wyjściowych neuronów ukrytych przez wagi
    net_input_out_neurons = np.sum(out_hidden_x_weights, axis=1)  # Sumowanie wartości wejściowych neuronów wyjściowych
    output_out_neurons = []

    for outputNeuron in range(0, 26):
        total_input_neuron = net_input_out_neurons[outputNeuron] + anomaly_output[outputNeuron]
        output_out_neurons.append(logistic(total_input_neuron))  # Obliczanie wartości wyjściowej neuronu wyjściowego

    return output_out_neurons, output_hidden_neurons

def calculateErrorAtOutput(output_out_neurons, target_output):
    output_error = []
    for output_neuron in range(0, 26):
        output_neuron_error = output_out_neurons[output_neuron] - target_output[output_neuron]  # Obliczanie błędu neuronu wyjściowego
        output_error.append(output_neuron_error)
    return output_error

def backPropagate(weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output, normalized, output_error, output_out_neurons, output_hidden_neurons, learning_rate,
                  momentum):
    old_weight_hidden_output = np.array(weight_hidden_output[:, :])
    old_weight_input_hidden = np.array(weight_input_hidden[:, :])

    [number_hidden_neurons, height, width] = weight_input_hidden.shape
    for output_neuron in range(0, 26):
        for hidden_neuron in range(0, number_hidden_neurons):
            adjustment = (learning_rate * output_error[output_neuron] * output_out_neurons[output_neuron] * (
                        1 - output_out_neurons[output_neuron]) * output_hidden_neurons[hidden_neuron])  # Obliczanie korekty wag
            weight_hidden_output[output_neuron, hidden_neuron] = (momentum * weight_hidden_output[
                output_neuron, hidden_neuron]) - adjustment

    for hidden_neuron in range(0, number_hidden_neurons):
        delta_total_error_hidden_neuron = 0
        for output_neuron in range(0, 26):
            delta_error_output_neuron = output_error[output_neuron] * output_out_neurons[output_neuron] * (
                        1 - output_out_neurons[output_neuron]) * old_weight_hidden_output[output_neuron, hidden_neuron]
            delta_total_error_hidden_neuron = delta_total_error_hidden_neuron + delta_error_output_neuron

        for i in range(0, height):
            for j in range(0, width):
                delta_total_error_input_neuron_weight = delta_total_error_hidden_neuron * output_hidden_neurons[
                    hidden_neuron] * (1 - output_hidden_neurons[hidden_neuron]) * normalized[i, j]  # Obliczanie korekty wag dla warstwy wejściowej
                weight_input_hidden[hidden_neuron, i, j] = (momentum * weight_input_hidden[hidden_neuron, i, j]) - (
                            learning_rate * delta_total_error_input_neuron_weight)

    return weight_input_hidden, weight_hidden_output

def trainNet(weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output, height, width, number_training_samples, learning_rate, momentum, target_error):
    iteration = 0
    total_error = 1
    error_list = []
    y_axis = []

    while total_error > target_error:

        for letter_train in range(0, 26):
            target_output = np.zeros(26)
            target_output[letter_train] = 1

            for n in range(0, number_training_samples):
                training_sample = 'Litery/%s/%s(%d).png' % (
                letters[letter_train].capitalize(), letters[letter_train], n + 1)  # Wczytywanie próbki treningowej

                character_in = im.open(training_sample)
                black_white = normal.convertToBW(character_in)  # Konwersja na obraz czarno-biały
                toggled_black_white = normal.toggleOnesAndZeros(black_white)  # Zamiana czarnych i białych pikseli
                cropped_black_white = normal.crop(toggled_black_white)  # Przycinanie obrazu
                normalized = normal.normalize(cropped_black_white, width, height)  # Normalizacja obrazu

                output_out_neurons, output_hidden_neurons = feedForward(normalized, weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output)  # Wykonanie propagacji w przód
                output_error = calculateErrorAtOutput(output_out_neurons, target_output)  # Obliczanie błędu wyjścia
                weight_input_hidden, weight_hidden_output = backPropagate(weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output, normalized, output_error, output_out_neurons,output_hidden_neurons, learning_rate,momentum)  # Wykonanie propagacji wstecznej

        total_error = 0
        for x in range(0, 26):
            squared = 0.5 * output_error[x] ** 2
            total_error = total_error + squared

        print('Aktualny stan błędu = %f' % total_error)
        iteration = iteration + 1
        error_list.append(total_error)
        y_axis.append(iteration)

    print('Końcowa liczba iteracji %d' % iteration)
    plt.plot(y_axis, error_list)
    plt.ylabel('Błąd docelowy')
    plt.xlabel('Liczba iteracji')
    plt.show()

    return (weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output)

def recognizeCharacter(input_normalized, weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output):
    output_out_neurons, outputOfHiddenNeurons = feedForward(input_normalized, weight_input_hidden, weight_hidden_output, anomaly_hidden, anomaly_output)  # Wykonanie propagacji w przód
    max_out = np.argmax(output_out_neurons)  # Znalezienie indeksu neuronu wyjściowego o największej wartości
    return letters[max_out]  # Zwrócenie odpowiedniej litery na podstawie indeksu

"""
initializeWeights: Inicjalizuje macierze wag oraz wektory anomałii dla sieci neuronowej.

logistic: Funkcja logistyczna, używana do obliczania wartości wyjściowej neuronów.

feedForward: Wykonuje propagację w przód, obliczając wartości wyjściowe neuronów.

calculateErrorAtOutput: Oblicza błąd wyjścia na podstawie wartości wyjściowych neuronów i oczekiwanych wyjść.

backPropagate: Wykonuje propagację wsteczną, aktualizując wagi na podstawie obliczonych błędów.

trainNet: Szkoli sieć neuronową na podstawie dostępnych danych treningowych.

recognizeCharacter: Rozpoznaje pojedynczy znak na podstawie podanego obrazu znormalizowanego.

"""