from flask import Flask
from flask import request,render_template,url_for
import pickle

app=Flask(__name__)

model = pickle.load(open('tree_gridcv.pkl', 'rb'))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	AQI_predict = None
	if request.method == 'POST':
		# field1 = request.form['field1']
		AQI_predict = model.predict([[
    float(request.form["T"]),
    float(request.form["TM"]),
    float(request.form["Tm"]),
    float(request.form["SLP"]),
    float(request.form["H"]),
    float(request.form["VV"]),
    float(request.form["V"]),
    float(request.form["VM"])
]])    
	
	return render_template('result.html', prediction=AQI_predict)

if __name__=='__main__':
	app.run()
