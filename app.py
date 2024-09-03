from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
data_file = "data/DATASET - Sheet1.csv"  
df = pd.read_csv(data_file)

# Preprocess data
def preprocess_data(row):
    return ' '.join([str(item) for item in row])

df['tags'] = df.apply(preprocess_data, axis=1)

# Vectorize text data
cv = CountVectorizer()
vector = cv.fit_transform(df['tags']).toarray()

# Compute cosine similarity
similarity = cosine_similarity(vector)

# Recommendation function
def recommend(input_data):
    input_vector = cv.transform([preprocess_data(input_data)]).toarray()
    distances = cosine_similarity(input_vector, vector)
    return distances

# Route to serve the index.html
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def get_recommendation():
    data = request.json
    input_data = data['input_data']
    results = recommend(input_data)
    water_requirements = df['WATER REQUIREMENT'][results.argmax()]
    return jsonify({'water_requirements': water_requirements})

if __name__ == '__main__':
    app.run(debug=True)
