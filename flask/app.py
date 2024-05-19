from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!"

@app.route('/save_data')
def save_data():
    return "Hello Save Data!"