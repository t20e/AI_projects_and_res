

/*

Basic Neural Network built in C++ from scratch
TODO

*/ 

// #include <iostream>
// #include <vector>
// #include <cmath>
// #include <cstdlib>
// #include <ctime>
// #include <numeric>


// // Sigmoid activation function
// double sigmoid(double x) {
//     return 1.0 / (1.0 + exp(-x));
// }

// // Derivative of the sigmoid function
// double sigmoid_derivative(double x) {
//     return x * (1.0 - x);
// }

// // A simple structure to represent a neuron
// struct Neuron {
//     double value;
//     double bias;
// };


// // Represents a layer of neurons in the network
// class Layer {
// public:
//     std::vector<Neuron> neurons;
//     std::vector<std::vector<double>> weights;

//     Layer(int num_neurons, int next_layer_size) {
//         neurons.resize(num_neurons);
        
//         // Initialize neurons with random biases
//         for (auto& neuron : neurons) {
//             neuron.bias = static_cast<double>(rand()) / RAND_MAX - 0.5; // Random value between -0.5 and 0.5
//         }

//         // If there's a next layer, initialize weights
//         if (next_layer_size > 0) {
//             weights.resize(num_neurons, std::vector<double>(next_layer_size));
//             for (int i = 0; i < num_neurons; ++i) {
//                 for (int j = 0; j < next_layer_size; ++j) {
//                     weights[i][j] = static_cast<double>(rand()) / RAND_MAX - 0.5; // Random value between -0.5 and 0.5
//                 }
//             }
//         }
//     }
// };


// // The main neural network class
// class NeuralNetwork {
// public:
//     std::vector<Layer> layers;
//     double learning_rate;

//     NeuralNetwork(const std::vector<int>& layer_sizes, double lr) : learning_rate(lr) {
//         // Create the layers based on the provided sizes
//         for (size_t i = 0; i < layer_sizes.size(); ++i) {
//             int next_layer_size = (i + 1 < layer_sizes.size()) ? layer_sizes[i + 1] : 0;
//             layers.emplace_back(layer_sizes[i], next_layer_size);
//         }
//     }

//     // Forward propagation
//     std::vector<double> feed_forward(const std::vector<double>& input) {
//         // Set the input layer's values
//         for (size_t i = 0; i < input.size(); ++i) {
//             layers[0].neurons[i].value = input[i];
//         }

//         // Propagate the signal through the network
//         for (size_t i = 0; i < layers.size() - 1; ++i) {
//             Layer& current_layer = layers[i];
//             Layer& next_layer = layers[i + 1];

//             // Calculate the value for each neuron in the next layer
//             for (size_t j = 0; j < next_layer.neurons.size(); ++j) {
//                 double sum = 0.0;
//                 for (size_t k = 0; k < current_layer.neurons.size(); ++k) {
//                     sum += current_layer.neurons[k].value * current_layer.weights[k][j];
//                 }
//                 next_layer.neurons[j].value = sigmoid(sum + next_layer.neurons[j].bias);
//             }
//         }

//         // Return the output layer's values
//         std::vector<double> output;
//         for (const auto& neuron : layers.back().neurons) {
//             output.push_back(neuron.value);
//         }
//         return output;
//     }

//     // Backpropagation to train the network
//     void back_propagate(const std::vector<double>& input, const std::vector<double>& target) {
//         // Perform a forward pass to get the outputs
//         std::vector<double> output = feed_forward(input);

//         // Calculate the error of the output layer
//         std::vector<double> output_deltas(output.size());
//         for (size_t i = 0; i < output.size(); ++i) {
//             output_deltas[i] = (target[i] - output[i]) * sigmoid_derivative(output[i]);
//         }

//         // Backpropagate the error through the hidden layers
//         for (int i = layers.size() - 2; i >= 0; --i) {
//             Layer& current_layer = layers[i];
//             Layer& next_layer = layers[i + 1];
//             std::vector<double>& deltas_next = (i == layers.size() - 2) ? output_deltas : std::vector<double>(); // Using a temporary vector for clarity
            
//             // Calculate the deltas for the current layer
//             std::vector<double> current_deltas(current_layer.neurons.size(), 0.0);
//             for (size_t j = 0; j < current_layer.neurons.size(); ++j) {
//                 double sum = 0.0;
//                 for (size_t k = 0; k < next_layer.neurons.size(); ++k) {
//                     sum += next_layer.weights[j][k] * deltas_next[k];
//                 }
//                 current_deltas[j] = sum * sigmoid_derivative(current_layer.neurons[j].value);
//             }
            
//             // Update weights and biases for the current layer
//             for (size_t j = 0; j < next_layer.neurons.size(); ++j) {
//                 for (size_t k = 0; k < current_layer.neurons.size(); ++k) {
//                     current_layer.weights[k][j] += learning_rate * deltas_next[j] * current_layer.neurons[k].value;
//                 }
//                 next_layer.neurons[j].bias += learning_rate * deltas_next[j];
//             }
//         }
//     }

//     // Train the network for a number of epochs
//     void train(const std::vector<std::vector<double>>& training_data, const std::vector<std::vector<double>>& targets, int epochs) {
//         for (int i = 0; i < epochs; ++i) {
//             double total_error = 0.0;
//             for (size_t j = 0; j < training_data.size(); ++j) {
//                 back_propagate(training_data[j], targets[j]);
//                 std::vector<double> output = feed_forward(training_data[j]);
//                 total_error += std::pow(targets[j][0] - output[0], 2);
//             }
//             if (i % 1000 == 0) {
//                 std::cout << "Epoch " << i << ", Total Error: " << total_error << std::endl;
//             }
//         }
//     }
// };

// int main() {
//     srand(static_cast<unsigned>(time(0)));

//     // Define the network architecture: 2 input neurons, 3 hidden neurons, 1 output neuron
//     NeuralNetwork nn({2, 3, 1}, 0.5);

//     // Training data for the XOR problem
//     std::vector<std::vector<double>> training_data = {
//         {0, 0},
//         {0, 1},
//         {1, 0},
//         {1, 1}
//     };
//     std::vector<std::vector<double>> targets = {
//         {0},
//         {1},
//         {1},
//         {0}
//     };

//     // Train the network
//     nn.train(training_data, targets, 10000);

//     // Test the network after training
//     std::cout << "\nTesting the trained network:" << std::endl;
//     for (size_t i = 0; i < training_data.size(); ++i) {
//         std::vector<double> output = nn.feed_forward(training_data[i]);
//         std::cout << "Input: {" << training_data[i][0] << ", " << training_data[i][1] << "} -> Expected Output: " << targets[i][0] << ", Actual Output: " << output[0] << std::endl;
//     }

//     return 0;
// }
// // The main neural network class
// class NeuralNetwork {
// public:
//     std::vector<Layer> layers;
//     double learning_rate;

//     NeuralNetwork(const std::vector<int>& layer_sizes, double lr) : learning_rate(lr) {
//         // Create the layers based on the provided sizes
//         for (size_t i = 0; i < layer_sizes.size(); ++i) {
//             int next_layer_size = (i + 1 < layer_sizes.size()) ? layer_sizes[i + 1] : 0;
//             layers.emplace_back(layer_sizes[i], next_layer_size);
//         }
//     }

//     // Forward propagation
//     std::vector<double> feed_forward(const std::vector<double>& input) {
//         // Set the input layer's values
//         for (size_t i = 0; i < input.size(); ++i) {
//             layers[0].neurons[i].value = input[i];
//         }

//         // Propagate the signal through the network
//         for (size_t i = 0; i < layers.size() - 1; ++i) {
//             Layer& current_layer = layers[i];
//             Layer& next_layer = layers[i + 1];

//             // Calculate the value for each neuron in the next layer
//             for (size_t j = 0; j < next_layer.neurons.size(); ++j) {
//                 double sum = 0.0;
//                 for (size_t k = 0; k < current_layer.neurons.size(); ++k) {
//                     sum += current_layer.neurons[k].value * current_layer.weights[k][j];
//                 }
//                 next_layer.neurons[j].value = sigmoid(sum + next_layer.neurons[j].bias);
//             }
//         }

//         // Return the output layer's values
//         std::vector<double> output;
//         for (const auto& neuron : layers.back().neurons) {
//             output.push_back(neuron.value);
//         }
//         return output;
//     }

//     // Backpropagation to train the network
//     void back_propagate(const std::vector<double>& input, const std::vector<double>& target) {
//         // Perform a forward pass to get the outputs
//         std::vector<double> output = feed_forward(input);

//         // Calculate the error of the output layer
//         std::vector<double> output_deltas(output.size());
//         for (size_t i = 0; i < output.size(); ++i) {
//             output_deltas[i] = (target[i] - output[i]) * sigmoid_derivative(output[i]);
//         }

//         // Backpropagate the error through the hidden layers
//         for (int i = layers.size() - 2; i >= 0; --i) {
//             Layer& current_layer = layers[i];
//             Layer& next_layer = layers[i + 1];
//             std::vector<double>& deltas_next = (i == layers.size() - 2) ? output_deltas : std::vector<double>(); // Using a temporary vector for clarity
            
//             // Calculate the deltas for the current layer
//             std::vector<double> current_deltas(current_layer.neurons.size(), 0.0);
//             for (size_t j = 0; j < current_layer.neurons.size(); ++j) {
//                 double sum = 0.0;
//                 for (size_t k = 0; k < next_layer.neurons.size(); ++k) {
//                     sum += next_layer.weights[j][k] * deltas_next[k];
//                 }
//                 current_deltas[j] = sum * sigmoid_derivative(current_layer.neurons[j].value);
//             }
            
//             // Update weights and biases for the current layer
//             for (size_t j = 0; j < next_layer.neurons.size(); ++j) {
//                 for (size_t k = 0; k < current_layer.neurons.size(); ++k) {
//                     current_layer.weights[k][j] += learning_rate * deltas_next[j] * current_layer.neurons[k].value;
//                 }
//                 next_layer.neurons[j].bias += learning_rate * deltas_next[j];
//             }
//         }
//     }

//     // Train the network for a number of epochs
//     void train(const std::vector<std::vector<double>>& training_data, const std::vector<std::vector<double>>& targets, int epochs) {
//         for (int i = 0; i < epochs; ++i) {
//             double total_error = 0.0;
//             for (size_t j = 0; j < training_data.size(); ++j) {
//                 back_propagate(training_data[j], targets[j]);
//                 std::vector<double> output = feed_forward(training_data[j]);
//                 total_error += std::pow(targets[j][0] - output[0], 2);
//             }
//             if (i % 1000 == 0) {
//                 std::cout << "Epoch " << i << ", Total Error: " << total_error << std::endl;
//             }
//         }
//     }
// };

// int main() {
//     srand(static_cast<unsigned>(time(0)));

//     // Define the network architecture: 2 input neurons, 3 hidden neurons, 1 output neuron
//     NeuralNetwork nn({2, 3, 1}, 0.5);

//     // Training data for the XOR problem
//     std::vector<std::vector<double>> training_data = {
//         {0, 0},
//         {0, 1},
//         {1, 0},
//         {1, 1}
//     };
//     std::vector<std::vector<double>> targets = {
//         {0},
//         {1},
//         {1},
//         {0}
//     };

//     // Train the network
//     nn.train(training_data, targets, 10000);

//     // Test the network after training
//     std::cout << "\nTesting the trained network:" << std::endl;
//     for (size_t i = 0; i < training_data.size(); ++i) {
//         std::vector<double> output = nn.feed_forward(training_data[i]);
//         std::cout << "Input: {" << training_data[i][0] << ", " << training_data[i][1] << "} -> Expected Output: " << targets[i][0] << ", Actual Output: " << output[0] << std::endl;
//     }

//     return 0;
// }