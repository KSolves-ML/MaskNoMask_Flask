import flask
from flask import Flask


app = Flask(__name__)

@app.route("/")
def predict():
    response = {}
    response["response"] = {
        "Name":"Gaurav",
        "Work":"AI Developer"
    }
    return flask.jsonify(response)

app.run()