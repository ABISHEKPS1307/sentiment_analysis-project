from flask import Flask, render_template, request
import pickle

# Load the trained model and vectorizer
with open('sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form['text']
    text_vectorized = vectorizer.transform([text])  # Transform the input text
    prediction = model.predict(text_vectorized)[0]  # Get the predicted sentiment
    return render_template('result.html', sentiment=prediction)

if __name__ == '__main__':
    app.run(debug=True)
