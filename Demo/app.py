import re
import pickle
import streamlit as st
from underthesea import word_tokenize


st.set_page_config(page_title='Vietnamese Legal Documents Classification', page_icon='😏',layout='wide', initial_sidebar_state='collapsed')
st.title('Vietnamese Legal Documents Classification')

# read Vietnamese stopwords
# =========================
stopword_file = "vietnamese-stopwords.txt"
with open(stopword_file, "r", encoding="utf-8") as f:
    stopword_content = f.read()
stop_words = stopword_content.splitlines()
# =========================

# load TF-IDF Vectorizer model
# =========================
vectorizer = pickle.load(open('./vectorizer.pkl', 'rb'))
# =========================

# load class_decoder models, MLP model
# =========================
model = pickle.load(open('./best_nn_model.pkl', 'rb'))
class_decoder = pickle.load(open('./class_encoder.pkl', 'rb'))
# =========================

document = st.text_area(label='Vietnamese legal document:', height=500)
btn_predict = st.button(label='Classify')

if btn_predict:
    # preprocessing text
    # =========================
    cleaned_content = document.lower()
    cleaned_content = re.sub('[^a-zA-Zà-ỹĂÂĐÊÔƠƯơư]', ' ', cleaned_content)
    cleaned_content = word_tokenize(cleaned_content)
    cleaned_content = [word for word in cleaned_content if word not in stop_words]
    cleaned_content = ' '.join(cleaned_content)
    cleaned_content = re.sub(' +', ' ', cleaned_content)
    # =========================
    # Vectorizer text
    input = vectorizer.transform([cleaned_content])
    # Get prediction
    output = model.predict(input)
    st.success(class_decoder.inverse_transform(output)[0])