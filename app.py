from flask import Flask, request, render_template
import pandas as pd
import pickle
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

app = Flask(__name__)

# Load models and vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf_vectorizer = pickle.load(f)

with open('PassiveAggressiveClassifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the drugs data
df1 = pd.read_csv('drugs1.tsv', sep='\t')
df2 = pd.read_csv('drugs2.tsv', sep='\t')
df = pd.concat([df1, df2], axis=0)

# Data Preprocessing
stop = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def cleanText(raw_review):
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    words = letters_only.lower().split()
    meaningful_words = [w for w in words if w not in stop]
    lemmatized_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    return ' '.join(lemmatized_words)

def top_drugs_extractor(condition):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'], ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_lst if drug_lst else ['No top drugs available']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        processed_text = cleanText(user_input)
        tfidf_input = tfidf_vectorizer.transform([processed_text])
        prediction = model.predict(tfidf_input)[0]

        # Mapping prediction to condition
        if prediction == "High Blood Pressure":
            target = "High Blood Pressure"
        elif prediction == "Depression":
            target = "Depression"
        elif prediction == "Diabetes, Type 2":
            target = "Diabetes, Type 2"
        else:
            target = "Birth Control"

        top_drugs = top_drugs_extractor(target)
        return render_template('index.html', condition=target, drugs=top_drugs)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
