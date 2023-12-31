from flask import Flask, jsonify, request
import os

import numpy as np


import pybaseball as pb

def get_player_id_number(last_name: str, first_name: str, player_num: int = 0) -> int:
    """
    Finds the player ID based on the player's last name and first name.
    Uses pybaseball's playerid_lookup function.

    Args:
        last_name (str): The last name of the player.
        first_name (str): The first name of the player.
        player_num (int, optional): The player number to specify which player to look at if there are multiple players with the same name. Defaults to 0.

    Returns:
        int: The player ID.

    Raises:
        ValueError: If the player ID lookup is empty.
    """

    try:
        return pb.playerid_lookup(last_name, first_name)["key_mlbam"][player_num]
    except KeyError:
        raise ValueError(
            "Player ID lookup failed. No player found with the given name."
        )



app = Flask(__name__)


@app.route("/")
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


@app.route("/random-int")
def random_int():
    random_number = np.random.randint(0, 100)  # Generates a random int between 0 and 99
    return jsonify({"random_integer": random_number})

@app.route(f"/player-id", methods=["GET"])
def get_player_id():
    last_name = request.args.get("last_name")
    first_name = request.args.get("first_name")
    player_num = request.args.get("player_num", default=0, type=int)  # Set default to 0

    print(f"Getting player ID for {first_name} {last_name}")

    if not last_name or not first_name:
        return jsonify({"error": "Missing last name or first name"}), 400

    try:
        player_id = get_player_id_number(last_name, first_name, player_num)
        print(f"Player ID: {player_id}")
        return jsonify({"player_id": int(player_id)}), 200
    except ValueError as e:
        print(str(e))
        return jsonify({"error": str(e)}), 404


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
