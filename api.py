from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('logistic_regression_model.pkl', 'rb') as f:
    clf = pickle.load(f)

app = Flask(__name__)
api = Api()
CORS(app)

@app.route('/api/post', methods=['POST'])
def handle_post_request():
    data = request.get_json()
    comment = data.get('comment')
    if comment:
        X_new_tfidf = tfidf_vectorizer.transform([comment])
        prediction = clf.predict(X_new_tfidf)
        prediction_array = prediction.tolist()
        if 1 in prediction_array[0]:
            return jsonify({"message" : 1})
        else:
            return jsonify({"message": 0}), 200  
    else:
        return jsonify({"message": "Comment is null"}), 401

if __name__ == '__main__':
    app.run(debug=True)