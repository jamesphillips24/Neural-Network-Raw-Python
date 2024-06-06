NeuralNetwork1 is a veryyyy early version. It's my first working NN and it's using almost exclusively raw python except numpy for euler's number and generating data. This is not a reflection of my coding ability, moreso of my NN understanding.
The code is horribly written, very inefficient, uncommented, and all around unpolished. I committed it as soon as it started functioning for the sake of getting my early version on here.
I used very inefficient coding practives in a lot of cases (the weights and biases initialization for example) for the sake of keeping it simple to retain my train of thought in regard to the actual NN structure.
It takes a single X coordinate and a Y coordinate from the funciton 1/2 * x^3, and learns to fit the 'curve' (only one point). Within 100 runs the loss is virtually zero.
It has one input, one hidden layer of 8 neurons, and one output. It uses a simple sigmoid activation function for the hidden layer (the output doesn't need an activation function as it isn't outputting something like a probability, and therefore
it doesn't need a bias either, as that scalar addition would be redundant). The loss is calculated using a simple mean square error with an added 1/2 multiplier to make the derivative a bit simpler. Below I'm gonna write some of the 
math to find the cost gradient as it's not very obvious in the code.

Forward Prop:

Hidden_layer_input(i) = Weight(1)(i) * input + Bias(i)
Hidden_layer_output(i) = sigmoid(Hidden_layer_input(i))
Output = sum(Weight(2)(i) * Hidden_layer_output(i))
Loss = 1/2(Output - expected_output)^2

Here, Weight(1)(i) is the ith weight in the first layer and Weight(2)(i) is the ith weight in the second layer. Everything else should be self explanatory where i is the ith neuron.

Back Prop:

dLoss/dWeight(2)(i) = dLoss/dOuput * dOutput/dWeight(2)(i) = (Output - expected_output) * Hidden_layer_output(i)
dLoss/dWeight(1)(i) = dLoss/dOutput * dOutput/dHidden_layer_output(i) * dHidden_layer_output(i)/dHidden_layer_input(i) * dHidden_layer_input(i)/dWeight(1)(i)
= (Output - expected_output) * Weight(2)(i) * (Hidden_layer_output(i) * (1 - Hidden_layer_output(i)) * input
dLoss/dBias(i) = dLoss/dOutput * dOutput/dHidden_layer_output(i) * dHidden_layer_output(i)/dHidden_layer_input(i) * dHidden_layer_input(i)/dBias(i)
= (Output - expected_output) * Weight(2)(i) * (Hidden_layer_output(i) * (1 - Hidden_layer_output(i))

Here, through math, the derivative of sigmoid(Hidden_layer_input(i)) can be shown to be equal to (Hidden_layer_output(i) * (1 - Hidden_layer_output(i))
