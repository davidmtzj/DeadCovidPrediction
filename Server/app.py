from flask import Flask, request, make_response, redirect, render_template, session, url_for, flash, jsonify, request
from flask_cors import CORS
import os
import joblib
import numpy as np

from app import Config
from app import create_app

#app = create_app()
app = Flask(__name__)


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
    #print("Request Data", request.json)

    covid_linear_SVC   = joblib.load('./models/covid_Linear_SVC_model.pkl')
    covid_SGD          = joblib.load('./models/covid_SGDClassifier_model.pkl')
    covid_forest_model = joblib.load('./models/covid_ExtraTreesClassifier_model.pkl')

    features = ['SEXO', 'INTUBADO', 'NEUMONIA', 'EDAD', 'EMBARAZO', 'DIABETES', 'EPOC',
       'ASMA', 'INMUSUPR', 'HIPERTENSION', 'OTRA_COM', 'CARDIOVASCULAR',
       'OBESIDAD', 'RENAL_CRONICA', 'TABAQUISMO', 'UCI']

    try:    

        for feature in features:
            #print(request.json['Features'][0][feature])
            X_features.append(int(request.json['Features'][0][feature]))

    except Exception as e:
        return f"An Error Occured: {e}"


    #X_test = np.array([2,0,0,27,0,0,0,0,0,0,0,0,0,0,0,0])
    X_test = np.array(X_features)
    X_test = X_test.reshape(1,-1)

    print("np array:",X_test)
    # ExtraTreesClassifier / Model
    prediction_forest  = covid_forest_model.predict(X_test) #1 row all colums
    probability_forest = covid_forest_model.predict_proba(X_test)
    print("Forest Prediction", prediction_forest)
    print("Features Forest...")
    print(covid_forest_model.feature_importances_)
    
    #Linear SVC
    prediction_svc  = covid_linear_SVC.predict(X_test) #1 row all colums
    print("SVC Prediction", prediction_svc)
    print("Features SVC...")
    print(covid_linear_SVC.coef_)

    #SGD
    prediction_sgd  = covid_SGD.predict(X_test) #1 row all colums
    print("SGD Prediction", prediction_sgd)

#    if prediction_svc[0] == 1 or prediction_forest[0] == 1 or prediction_sgd[0] == 1:
        
    if  prediction_sgd[0] == 1:
        message = "You will die! Please stay Home :)"
        prediction = 1
    else:
        message = "You will not die, but Please stay Home :)"


    return jsonify({
        'prediction' : float(prediction_sgd),
        'message'    : message
    }), 200

port = int(os.environ.get('PORT', 5000))
if __name__ == '__main__':
    app.config.from_object(Config)
    app.run(port=port)
    
