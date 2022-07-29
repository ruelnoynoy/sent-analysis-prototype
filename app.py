from flask import Flask, render_template,request
import numpy as np
from model import svmmodel, nbmodel, tf

app = Flask(__name__)
result = ""
messages = []

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        textContent = request.form['textContent']

        customer_reviews_array = np.array([textContent])
        customer_review_vector = tf.transform(customer_reviews_array)

        if nbmodel.predict(customer_review_vector) == 1:
            result = "Positive Sentiment"
        else:
            result = "Negative Sentiment"
        return render_template('index.html', result = result)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)