from flask import Flask
import json

app = Flask(__name__)

# Function to load data from JSON file
def load_json_data(filename):
    with open(filename) as file:
        return json.load(file)

# Load data once when the app starts
json_data = load_json_data('tensorflowjs_files/model.json')

@app.route('/')
def hello_world():
    # Use data from JSON file
    return json_data

if __name__ == '__main__':
    app.run()