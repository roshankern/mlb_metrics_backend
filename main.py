from flask import Flask, jsonify
import os

import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


@app.route("/random-int")
def random_int():
    random_number = np.random.randint(0, 100)  # Generates a random int between 0 and 99
    return jsonify({"random_integer": random_number})


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
