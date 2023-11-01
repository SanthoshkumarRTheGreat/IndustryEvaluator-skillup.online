import random
import json
import torch
from flask import Flask, request, jsonify, render_template
from long_responses import NeuralNet
from chatbot import bag_of_words, tokenize, stem
import os

app = Flask(__name__)

# Load chatbot data
os.chdir('C:/Users/arara/OneDrive/Desktop/chatbot/')
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)
input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
model_state = data["model_state"]
tags = data["tags"]

model = NeuralNet(input_size, hidden_size, output_size)
model.load_state_dict(model_state)
model.eval()

bot_name = "zoro"

# Route for rendering the chat interface
@app.route('/')
def chat_interface():
    os.chdir('C:/Users/arara/OneDrive/Desktop/chatbot/')
    return render_template('index.html')

# Function to handle chatbot responses
def get_chatbot_response(user_input):
    sentence = tokenize(user_input)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    else:
        return "I do not understand... please only ask queries related to food delivery."

# Route for handling chat requests
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    chatbot_response = get_chatbot_response(user_input)
    return jsonify({'response': chatbot_response})

if __name__ == '__main__':
    app.run(debug=True)
