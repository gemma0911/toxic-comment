from flask import Flask, request, jsonify
from flask_restful import Api
from flask_cors import CORS
import pickle
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

with open('train/tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)
with open('train/logistic_regression_model.pkl', 'rb') as f:
    clf = pickle.load(f)

app = Flask(__name__)
api = Api()
CORS(app)

def data_processing(data):
    data = data.lower()
    data = re.sub(r'[^a-zA-Z\s]', '', data)
    words = word_tokenize(data)
    stop_words = set(ENGLISH_STOP_WORDS)
    words = [word for word in words if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    processed_data = ' '.join(words)
    return processed_data


@app.route('/api/toxic', methods=['POST'])
def toxic():
    try:
        data = request.get_json()
        comment = data.get('comment')
        comment = data_processing(comment)
        if comment:
            X_new_tfidf = tfidf_vectorizer.transform([comment])
            
            prediction = clf.predict(X_new_tfidf)
            prediction_array = prediction.tolist()
            
            if 1 in prediction_array[0]:
                return jsonify({"message": 1})
            else:
                return jsonify({"message": 0}), 200
        else:
            return jsonify({"message": "Comment is null"}), 401
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)}), 500


@app.route('/api/toxic_detail', methods=['POST'])
def toxic_detail():
    try:
        data = request.get_json()
        comment = data.get('comment')
        if not comment:
            return jsonify({"message": "Comment is null"}), 401
        
        comment = data_processing(comment)
        X_new_tfidf = tfidf_vectorizer.transform([comment])
        
        prediction = clf.predict(X_new_tfidf)
        prediction_array = prediction.tolist()
    
        return jsonify({
            "message": prediction_array,
        }), 200
    
    except Exception as e:
        return jsonify({"message": "An error occurred", "error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)