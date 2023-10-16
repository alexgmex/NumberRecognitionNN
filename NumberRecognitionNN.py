# Import libraries
import pandas as pd
import numpy as np
import pygame

# Establish initial variables
imgSize = 28
imgScaler = 10
progressBarWidth = 10
dataNormalisationValue = 255

training = "MNIST_Data/mnist_train.csv"
testing = "MNIST_Data/mnist_test.csv"
layers = [784,16,16,10]
learningRate = 0.1

########################################## FUNCTIONS ##########################################

# Function for preprocessing data, extracting from CSV and splitting into input and expected outputs
def preprocessData(trainingFilename,testingFilename,normalisationValue):
    # Make numpy array from CSV
    trainingRaw = pd.read_csv(trainingFilename,header=None).to_numpy()
    testingRaw = pd.read_csv(testingFilename,header=None).to_numpy()
    # Split first column from data
    trainingAnswers = trainingRaw[:,0] # First column only
    trainingData = trainingRaw[:,1:] # Second column:end
    testingAnswers = testingRaw[:,0] # First column only
    testingData = testingRaw[:,1:] # Second column:end
    # Normalise input data
    trainingData = trainingData/normalisationValue
    testingData = testingData/normalisationValue
    # Get data counts
    trainingExamples = np.size(trainingData,0)
    testingExamples = np.size(testingData,0)
    # Return all
    return trainingAnswers, trainingData, trainingExamples, testingAnswers, testingData, testingExamples

# Make visual representation
def animateTraining(imgDimension,imgData,imgAnswer,currentExample,totalExamples,normalisationValue):
    # Print image based off pixel brightness values
    for y in range(imgDimension):
        for x in range(imgDimension):
            currentPixel = normalisationValue*imgData[y*imgSize + x] # Grid to linear conversion, also denormalise it for the purposes of visual representation
            pygame.draw.rect(screen, (currentPixel, currentPixel, currentPixel), (x*imgScaler, y*imgScaler, imgScaler, imgScaler)) # Draw the image, but scaled up
    
    # Add "real" image 
    screen.blit(pygame.image.load("outputs/" + str(imgAnswer) + ".png"), (imgSize*imgScaler, 0))

    # Add progress bar
    progress = round((currentExample/totalExamples)*imgSize*imgScaler*2) # Progress percentage multiplied by the width of the window (fits progress bar to window)
    pygame.draw.rect(screen, (0, 255, 0), (0, imgSize*imgScaler + progressBarWidth, progress, progressBarWidth))

    # Update display
    pygame.display.update()

# Function for creating arrays of weights, biases and neurons
def initialiseLayers(layerFormat):

    # Set them up as lists (not numpy arrays as they will have different dimensions per layer)
    neurons = [0] # Leave an extra element in neurons so that we can shove in the input layer later
    deltas = [0] # Deltas same as neurons
    weights = []
    biases = []
    
    # Neurons and deltas default to zero, weights and biases are picked randomly from [-1,1)
    for i in range(1,len(layerFormat)):
        neurons.append(0)
        deltas.append(0)
        weights.append((2*(np.random.rand(layerFormat[i],layerFormat[i-1]))-1))
        biases.append((2*(np.random.rand(layerFormat[i],1))-1))
    
    return neurons, weights, biases, deltas


# Function for calculating the log-sigmoid activation function
def vectorSigmoid(value):
    return ((1/(1+np.exp(-value))))

######################################### //FUNCTIONS #########################################

# Initialise pygame
pygame.init()
screen = pygame.display.set_mode([imgSize*imgScaler*2, imgSize*imgScaler + 3*progressBarWidth])

# Preprocess data and initialise layers
trainingAnswers, trainingData, trainingExamples, testingAnswers, testingData, testingExamples = preprocessData(training,testing,dataNormalisationValue)
neurons, weights, biases, deltas = initialiseLayers(layers)


# Train NN
for z in range(trainingExamples):

    # Push the current training data into the first neuron layer (transpose to make into a 2d vertical column vector)
    neurons[0] = np.atleast_2d(trainingData[z]).T

    # Forward propagation
    for i in range(len(weights)):
        neurons[i+1] = vectorSigmoid(np.dot(weights[i],neurons[i]) + biases[i])
    

    # Back propagation of output layer
    desired = np.zeros((layers[-1],1))
    desired[trainingAnswers[z]] = 1
    deltas[-1] = (desired-neurons[-1])*neurons[-1]*(1-neurons[-1])

    biases[-1] += learningRate*deltas[-1]
    weights[-1] += learningRate*(neurons[-2].T)*deltas[-1]

    

    # Back propagation of hidden layers
    for i in range(len(weights)-1): # -1 because we have already done the last layer
        currentLayer = -2-i # working backwards from the 2nd last layer
        deltas[currentLayer] = neurons[currentLayer]*(1-neurons[currentLayer])*np.dot(weights[currentLayer+1].T,deltas[currentLayer+1])
        biases[currentLayer] += learningRate*deltas[currentLayer]
        weights[currentLayer] += learningRate*neurons[currentLayer-1].T*deltas[currentLayer]

    # Normalise output layer
    maxX = np.max(neurons[-1])
    minX = np.min(neurons[-1])
    neurons[-1] = (neurons[-1]-minX)/(maxX-minX)
        
    # Print progress
    print(neurons[-1])
    print("I think this is a " + str(np.argmax(neurons[-1])) + "!" + "\n")

    # Finally draw the animation of the current training example
    animateTraining(imgSize,trainingData[z],trainingAnswers[z],z,trainingExamples,dataNormalisationValue)

    # Quit on window close
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()





# Test NN
amountCorrect = 0
for z in range(testingExamples):

    # Push the current training data into the first neuron layer (transpose to make into a 2d vertical column vector)
    neurons[0] = np.atleast_2d(testingData[z]).T

    # Forward propagation
    for i in range(len(neurons)-1):
        neurons[i+1] = vectorSigmoid(np.dot(weights[i],neurons[i]) + biases[i])
    
    currentGuess = np.argmax(neurons[-1])

    # Print output
    print("I think this is a " + str(currentGuess) + "! ")

    if testingAnswers[z] == currentGuess:
        print("(Correct!)\n")
        amountCorrect += 1
    else:
        print("(Wrong)\n")

print("Total accuracy: " + str(round(100*(amountCorrect/testingExamples))) + "%")