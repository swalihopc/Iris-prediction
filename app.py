from flask import Flask,render_template,request
app=Flask(__name__)
import numpy as np
import pickle
@app.route('/')
def home():
    return render_template('iris.html',prediction_text='')

@app.route('/Prediction',methods=['post'])
def predict():
    sl=float(request.form['sepal_Length'])
    sw=float(request.form['sepal_width'])
    pl=float(request.form['petal_length'])
    pw=float(request.form['petal_width'])
    
    newdata=np.array([[sl,sw,pl,pw]]) 
    model=pickle.load(open("model.pkl","rb"))  
    result=model.predict(newdata)
    if result == 0:
     prediction_text="setosa"
    elif result == 1:
     prediction_text="versicolor"
    elif result == 2:
     prediction_text="virginica"      
    else:
     prediction_text="Error" 
    return render_template('iris.html',prediction_text=prediction_text)
if __name__=="__main__":
    app.run(host='0.0.0.0',port=5000,debug=True)