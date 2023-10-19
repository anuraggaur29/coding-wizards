from flask import Flask, render_template, request
import nltk
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# Load NLTK data (uncomment if not already downloaded)
# nltk.download('punkt')
# nltk.download('stopwords')

fake = pd.read_csv('Fake.csv')
true = pd.read_csv('True.csv')

fake['target'] = 0
true['target'] = 1

data = pd.concat([fake, true], axis=0)
data = data.reset_index(drop=True)

data = data.drop(['subject', 'date', 'title'], axis=1)

stemmer = SnowballStemmer('english')
stop_words = stopwords.words('english')

def preprocess_text(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    filtered_tokens = [token for token in stemmed_tokens if token not in stop_words]
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

data['text'] = data['text'].apply(preprocess_text)

cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(data['text']).toarray()
y = data['target'].values

gnb = GaussianNB()
gnb.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    new_text = request.form['news_text']
    preprocessed_text = preprocess_text(new_text)
    X_new = cv.transform([preprocessed_text]).toarray()
    y_pred_new = gnb.predict(X_new)

    if y_pred_new[0] == 0:
        result = "Fake News"
    else:
        result = "True News"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
