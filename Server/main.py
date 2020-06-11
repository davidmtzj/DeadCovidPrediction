from flask import Flask, request, make_response, redirect, render_template, session, url_for, flash, jsonify
from flask_cors import CORS
import os
import joblib
import numpy as np

from app import Config
from app import create_app
from ml.mlModels import MLDeadPredictor

#app = create_app()
app = Flask(__name__)
app.config.from_object(Config)

CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/test')
def index():
    return jsonify({
        'path' : '/predictDeadCovid',
        'Description' : 'Covid Dead Prediction',
        'body_example' : '{            Features : [{                 SEXO: 2,                INTUBADO:1,                NEUMONIA:1,                EDAD:54,                EMBARAZO:0,                DIABETES:0,                EPOC:0,                ASMA:0,                INMUSUPR:0,                HIPERTENSION:1,                OTRA_COM:0,                CARDIOVASCULAR:0,                OBESIDAD:0,                RENAL_CRONICA:0,                TABAQUISMO:0,                UCI:0            }]        }'
    }), 200


@app.route('/', defaults={'path': ''}) 
@app.route('/<path:path>')
def dender_vue(path):
    return render_template("index.html")


@app.route('/predictDeadCovid', methods=['GET', 'POST'])
def predict_dead_covid():
    """
    Example values 
    SEXO               1
    INTUBADO           2
    NEUMONIA           2
    EDAD              44
    EMBARAZO           2
    DIABETES           2
    EPOC               2
    ASMA               2
    INMUSUPR           2
    HIPERTENSION       2
    OTRA_COM           2
    CARDIOVASCULAR     2
    OBESIDAD           2
    RENAL_CRONICA      2
    TABAQUISMO         2
    UCI                2
    
    
    """
    
    X_features = []

    features = ['SEXO', 'INTUBADO', 'NEUMONIA', 'EDAD', 'EMBARAZO', 'DIABETES', 'EPOC',
       'ASMA', 'INMUSUPR', 'HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR',
       'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 'UCI']

    try:    

        for feature in features:
            X_features.append(int(request.json['Features'][0][feature]))

    except Exception as e:
        return f"An Error Occured: {e}"

    predictor = MLDeadPredictor( X_features )
    prediction = predictor.predictSGD( )
        
    if  prediction[0] == 1:
        message = "You will die! Please stay Home :)"
        prediction = 1
    else:
        message = "You will not die, but Please stay Home :)"

    return jsonify({
        'prediction' : int(prediction),
        'message'    : message
    }), 200

    
