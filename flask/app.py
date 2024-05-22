from flask import Flask, request, jsonify
import tensorflow as tf

app = Flask(__name__)

@app.route("/")
def home():
    return "Hello, Flask!: {}".format(tf.__version__)

@app.route('/save_data')
def save_data():
    return "Hello Save Data!"