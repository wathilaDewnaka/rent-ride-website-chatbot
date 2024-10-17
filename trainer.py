import pickle
import random
import numpy as np
import json
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Initialize lemmatizer and load intents
lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())

words = []
classes = []
documents = []
ignore_characters = ['?', '!', '.', ',']

# Process intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Preprocess words and classes
words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_characters]
words = sorted(set(words))
classes = sorted(set(classes))

# Save words and classes using pickle
with open("words.pkl", "wb") as f:
    pickle.dump(words, f)
with open("classes.pkl", "wb") as f:
    pickle.dump(classes, f)

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in document[0]]

    for word in words:
        bag.append(1 if word in word_patterns else 0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append(bag + output_row)

random.shuffle(training)
training = np.array(training)

print(training)
trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# Define the TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax')
])

# Compile the model
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model
hist = model.fit(trainX, trainY, epochs=200, batch_size=5, verbose=1)

# Save the model
model.save('chatbot_model.h5')

print('Done')
