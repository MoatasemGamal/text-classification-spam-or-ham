labels = ['ham', 'spam']
maxlen = 77
vocab_size = 20000  # Only consider the top 20k words

import keras
from keras import ops
from keras import layers

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        return self.layernorm2(out1 + ffn_output)
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = ops.shape(x)[-1]
        positions = ops.arange(start=0, stop=maxlen, step=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    

embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 32  # Hidden layer size in feed forward network inside transformer

inputs = layers.Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.1)(x)
x = layers.Dense(20, activation="relu")(x)
x = layers.Dropout(0.1)(x)
outputs = layers.Dense(2, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs)
model.load_weights('transformer.keras')


from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.text import Tokenizer 
from tensorflow.keras.utils import pad_sequences

tokenizer_path = 'tokenizer.json'
with open(tokenizer_path, 'r', encoding='utf-8') as f:
    loaded_tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(loaded_tokenizer_json)

import re
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk import PorterStemmer
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


ps = PorterStemmer()

en_stopwords = set(stopwords.words("english"))
def remove_stopwords(text):
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in en_stopwords]
    return ' '.join(filtered_text)

def clean_text(text):
    text  = text.lower()
    text  = re.sub('[^a-zA-Z!\?]', ' ', text)
    text  = re.sub('\s{2,}', ' ', text)
    text  = remove_stopwords(text)
    words = [ps.stem(word) for word in word_tokenize(text)]
    text  = TreebankWordDetokenizer().detokenize(words)
    return text

def tokenize_text(text):
    return np.array(pad_sequences(tokenizer.texts_to_sequences([clean_text(text)]), maxlen=maxlen, padding='post', truncating='post'))

def predict(text):
    return labels[np.argmax(model(tokenize_text(text))[0])]


from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_message():
    message = request.form.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Message should not be empty.'})
    if len(message.split()) > 77:
        return jsonify({'error': 'Message should not exceed 77 words.'})

    result = predict(message)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
