## Kode Lengkap: `deteksi_ujaran_kebencian.py`

```Py
# =========================================
# DETEKSI UJARAN KEBENCIAN DENGAN DEEP LEARNING (LSTM)
# =========================================

# Langkah 1: Import Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split

import nltk
import string
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud

import tensorflow as tf
from tensorflow import keras
from keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Unduh resource nltk
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')


# =========================================
# Langkah 2: Memuat Dataset
# =========================================
df = pd.read_csv('hate_speech.csv')
print(df.head())
print(df.shape)
print(df.info())

plt.pie(df['class'].value_counts().values,
        labels=df['class'].value_counts().index,
        autopct='%1.1f%%')
plt.title("Distribusi Kelas Asli")
plt.show()

# =========================================
# Langkah 3: Menyeimbangkan Dataset
# =========================================
class_0 = df[df['class'] == 0]  # Hate Speech
class_1 = df[df['class'] == 1].sample(n=3500, random_state=42)  # Offensive
class_2 = df[df['class'] == 2]  # Neutral

balanced_df = pd.concat([class_0, class_0, class_0, class_1, class_2], axis=0)

plt.pie(balanced_df['class'].value_counts().values,
        labels=balanced_df['class'].value_counts().index,
        autopct='%1.1f%%')
plt.title("Balanced Class Distribution")
plt.show()


# =========================================
# Langkah 4: Pra-pemrosesan Teks
# =========================================
balanced_df['tweet'] = balanced_df['tweet'].str.lower()

punctuations_list = string.punctuation
def remove_punctuations(text):
    temp = str.maketrans('', '', punctuations_list)
    return text.translate(temp)

balanced_df['tweet'] = balanced_df['tweet'].apply(lambda x: remove_punctuations(x))

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return " ".join(words)

balanced_df['tweet'] = balanced_df['tweet'].apply(preprocess_text)

# Visualisasi WordCloud
def plot_word_cloud(data, typ):
    corpus = " ".join(data['tweet'])
    wc = WordCloud(max_words=100, width=800, height=400, collocations=False).generate(corpus)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"Word Cloud for {typ} Class", fontsize=15)
    plt.show()

plot_word_cloud(balanced_df[balanced_df['class'] == 2], typ="Neutral")


# =========================================
# Langkah 5: Tokenisasi dan Padding
# =========================================
features = balanced_df['tweet']
target = balanced_df['class']

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# One-hot encode label
Y_train = pd.get_dummies(Y_train)
Y_val = pd.get_dummies(Y_val)

max_words = 5000
max_len = 100
tokenizer = Tokenizer(num_words=max_words, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

X_train_padded = pad_sequences(X_train_seq, maxlen=max_len, padding='post', truncating='post')
X_val_padded = pad_sequences(X_val_seq, maxlen=max_len, padding='post', truncating='post')


# =========================================
# Langkah 6: Membangun Model LSTM
# =========================================
model = keras.models.Sequential([
    layers.Embedding(input_dim=max_words, output_dim=32, input_length=max_len),
    layers.Bidirectional(layers.LSTM(16)),
    layers.Dense(512, activation='relu', kernel_regularizer='l1'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(3, activation='softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()


# =========================================
# Langkah 7: Melatih Model
# =========================================
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(patience=2, monitor='val_loss', factor=0.5, verbose=1)

history = model.fit(X_train_padded, Y_train,
                    validation_data=(X_val_padded, Y_val),
                    epochs=50,
                    batch_size=32,
                    callbacks=[es, lr])


# =========================================
# Langkah 8: Evaluasi Model
# =========================================
history_df = pd.DataFrame(history.history)

history_df[['loss', 'val_loss']].plot(title="Loss")
history_df[['accuracy', 'val_accuracy']].plot(title="Accuracy")
plt.show()

test_loss, test_acc = model.evaluate(X_val_padded, Y_val)
print(f"Validation Accuracy: {test_acc:.2f}")
```

## Instal library yang dibutuhkan

```py
pip install numpy pandas matplotlib seaborn nltk wordcloud tensorflow
```

## Jalankan program

```py
python deteksi_ujaran_kebencian.py
```

# OUTPUT

![gambar](foto/Figure_1.png)
![gambar](foto/Figure_2.png)
![gambar](foto/Figure_3.png)
![gambar](foto/Figure_2.1.png)
![gambar](foto/Figure_2.2.png)
