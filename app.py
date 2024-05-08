from flask import Flask,request,render_template
import numpy as np
import pandas
import sklearn
import pickle

# Importar los modelos
svm_model = pickle.load(open('svm_model.pkl','rb'))
label_encoders = pickle.load(open('label_encoders.pkl','rb'))
ms = pickle.load(open('min_max_scaler.pkl','rb'))

# crear flask
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict",methods=['POST'])
def predict():
    sl_no = request.form['sl_no']
    gender = request.form['gender']
    ssc_p = request.form['ssc_p']
    hsc_p = request.form['hsc_p']
    degree_p = request.form['degree_p']
    workex = request.form['workex']
    etest_p = request.form['etest_p']
    specialisation = request.form['specialisation']
    mba_p = request.form['mba_p']

    feature_list = [sl_no, gender, ssc_p, hsc_p, degree_p, workex, etest_p, specialisation, mba_p]
    
    #single_pred = np.array(feature_list).reshape(1, -1)
    #scaled_features = ms.transform(single_pred)
    #final_features = sc.transform(scaled_features)
    #prediction = model.predict(final_features)
    gender_encoded=label_encoders['gender'].transform([gender])[0]
    workex_encoded=label_encoders['workex'].transform([workex])[0]
    specialisation_encoded = label_encoders['specialisation'].transform([specialisation])[0]
    data = [[sl_no, gender_encoded, ssc_p, hsc_p, degree_p, workex_encoded, etest_p, specialisation_encoded, mba_p]]
    data_scaled = ms.transform(data)
    prediction = svm_model.predict(data_scaled)

    diccionario = {1: "Contratado", 0: "No Contratado"}

    if prediction[0] in diccionario:
        crop = diccionario[prediction[0]]
        result =("La persona sera: {} ".format(crop))
    else:
        result =("Sorry, no se tiene respuesta")
    return render_template('index.html',result = result)




# python main
if __name__ == "__main__":
    app.run(debug=True)