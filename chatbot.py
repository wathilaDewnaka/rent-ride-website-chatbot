import random
import json
import pickle
import numpy as np
import nltk
from flask_cors import CORS
import time
import requests

from flask import Flask, request, jsonify
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize Flask app
app = Flask(__name__)
CORS(app)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def get_order_status(order_id):
    api_url = f"http://localhost:8080/api/customer/bookings/get/" + order_id
    
    try:
        response = requests.get(api_url)
        
        if response.status_code == 200:
            print(response)
            data = response.json()
            return data.get("bookACarStatus", "Status not available")
        else:
            return "Error fetching order status"

    except:
        return "Error fetching order status"

count = 0

# API Route for chatbot response
@app.route("/chatbot", methods=['POST'])
def chatbot_response():
    global count
    message = request.json.get("message")
    if message:
        ints = predict_class(message)
        print(ints)
        
        tag = ints[0]['intent']

        # Check for order_status intent
        if tag == 'order_status' or count == 1:
            if count != 0:
                # Extract order ID from message
                # For simplicity, assume the message is just the order ID
                order_id = message.strip()  # Simplified extraction
                status = get_order_status(order_id)

                if status == "null":
                    response = get_response(ints, intents)
                else:
                    response = f"Your order status is: {status}"
                    count = 0
            
            else:
                response = get_response(ints, intents)
                count += 1
        else:
            response = get_response(ints, intents)
        
        time.sleep(1)
        return jsonify({"response": response})
    else:
        return jsonify({"error": "No message provided"}), 400

if __name__ == "__main__":
    print("Chatbot API is running!")
    app.run(debug=True)