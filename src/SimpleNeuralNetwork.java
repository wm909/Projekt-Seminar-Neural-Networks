import java.util.Arrays;
import java.util.Random;

public class SimpleNeuralNetwork {
    private int inputNodes;
    private int hiddenNodes;
    private int outputNodes;

    private double[][] weightsInputHidden;
    private double[][] weightsHiddenOutput;

    private double[] hiddenLayer;
    private double[] outputLayer;

    private double learningRate = 0.1;

    // Konstruktor
    public SimpleNeuralNetwork(int input, int hidden, int output) {
        this.inputNodes = input;
        this.hiddenNodes = hidden;
        this.outputNodes = output;

        // Initialisiere Gewichte mit zufälligen Werten
        weightsInputHidden = new double[inputNodes][hiddenNodes];
        weightsHiddenOutput = new double[hiddenNodes][outputNodes];

        initializeWeights();
    }

    // Initialisiere Gewichte mit zufälligen Werten zwischen -1 und 1
    private void initializeWeights() {
        Random rand = new Random();

        for (int i = 0; i < inputNodes; i++) {
            for (int j = 0; j < hiddenNodes; j++) {
                weightsInputHidden[i][j] = rand.nextDouble() * 2 - 1;
            }
        }

        for (int i = 0; i < hiddenNodes; i++) {
            for (int j = 0; j < outputNodes; j++) {
                weightsHiddenOutput[i][j] = rand.nextDouble() * 2 - 1;
            }
        }
    }

    // Aktivierungsfunktion: Sigmoid
    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Ableitung der Sigmoid-Funktion
    private double sigmoidDerivative(double x) {
        return x * (1 - x);
    }

    // Vorwärtspropagation
    public double[] feedForward(double[] inputs) {
        // Hidden Layer berechnen
        hiddenLayer = new double[hiddenNodes];
        for (int j = 0; j < hiddenNodes; j++) {
            double sum = 0;
            for (int i = 0; i < inputNodes; i++) {
                sum += inputs[i] * weightsInputHidden[i][j];
            }
            hiddenLayer[j] = sigmoid(sum);
        }

        // Output Layer berechnen
        outputLayer = new double[outputNodes];
        for (int j = 0; j < outputNodes; j++) {
            double sum = 0;
            for (int i = 0; i < hiddenNodes; i++) {
                sum += hiddenLayer[i] * weightsHiddenOutput[i][j];
            }
            outputLayer[j] = sigmoid(sum);
        }

        return outputLayer;
    }

    // Backpropagation-Trainingsmethode
    public void train(double[] inputs, double[] targets) {
        // Vorwärtspropagation
        feedForward(inputs);

        // Berechne Ausgabefehler
        double[] outputErrors = new double[outputNodes];
        for (int i = 0; i < outputNodes; i++) {
            outputErrors[i] = targets[i] - outputLayer[i];
        }

        // Berechne Ausgabe-Gradients
        double[] outputGradients = new double[outputNodes];
        for (int i = 0; i < outputNodes; i++) {
            outputGradients[i] = outputErrors[i] * sigmoidDerivative(outputLayer[i]);
        }

        // Berechne Hidden-Ausgabe-Gewichtsänderungen
        for (int i = 0; i < hiddenNodes; i++) {
            for (int j = 0; j < outputNodes; j++) {
                double delta = outputGradients[j] * hiddenLayer[i];
                weightsHiddenOutput[i][j] += learningRate * delta;
            }
        }

        // Berechne Hidden-Fehler
        double[] hiddenErrors = new double[hiddenNodes];
        for (int i = 0; i < hiddenNodes; i++) {
            double error = 0;
            for (int j = 0; j < outputNodes; j++) {
                error += outputErrors[j] * weightsHiddenOutput[i][j];
            }
            hiddenErrors[i] = error;
        }

        // Berechne Hidden-Gradients
        double[] hiddenGradients = new double[hiddenNodes];
        for (int i = 0; i < hiddenNodes; i++) {
            hiddenGradients[i] = hiddenErrors[i] * sigmoidDerivative(hiddenLayer[i]);
        }

        // Berechne Input-Hidden-Gewichtsänderungen
        for (int i = 0; i < inputNodes; i++) {
            for (int j = 0; j < hiddenNodes; j++) {
                double delta = hiddenGradients[j] * inputs[i];
                weightsInputHidden[i][j] += learningRate * delta;
            }
        }
    }

    // Main-Methode zum Testen des Netzes
    public static void main(String[] args) {
        // Beispiel: XOR-Problem
        SimpleNeuralNetwork nn = new SimpleNeuralNetwork(2, 4, 1);

        // Trainingsdaten für XOR
        double[][] inputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };

        double[][] targets = {
                {0},
                {1},
                {1},
                {0}
        };

        // Training
        for (int epoch = 0; epoch < 10000; epoch++) {
            for (int i = 0; i < inputs.length; i++) {
                nn.train(inputs[i], targets[i]);
            }
        }

        // Vorhersage testen
        System.out.println("Testergebnisse:");
        for (double[] input : inputs) {
            double[] output = nn.feedForward(input);
            System.out.printf("%s -> %.4f%n", Arrays.toString(input), output[0]);
        }
    }
}