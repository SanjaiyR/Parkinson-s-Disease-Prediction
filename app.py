from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load('parkinsons_model.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        prediction = model.predict([features])[0]
        if prediction == 1:
            result = "High likelihood of Parkinson's disease."
        else:
            result = "Low likelihood of Parkinson's disease."
        return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
