import streamlit as st
from ml_model import load_dataset, train_model
import base64

st.set_page_config(
    page_title="Spam Classifier",
    layout="centered"
)

# Background loader
def add_bg_from_local(image_file):
    with open(image_file, "rb") as file:
        encoded_string = base64.b64encode(file.read()).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}

    .main-title {{
        font-size: 38px;
        color: #ffffff;
        text-shadow: 1px 1px 4px #000;
        text-align: center;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }}

    /* Card background */
    .card {{
        background: rgba(255, 255, 255, 0.25);
        padding: 20px;
        border-radius: 12px;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
    }}

    /* Text area */
    textarea {{
        background: rgba(255,255,255,0.85) !important;
        border-radius: 10px !important;
    }}

    /* Soft alert colors */
    .spam-box {{
        background: rgba(255, 120, 120, 0.85);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-size: 20px;
        text-align: center;
        font-weight: bold;
    }}

    .ham-box {{
        background: rgba(120, 220, 120, 0.85);
        padding: 15px;
        border-radius: 10px;
        color: white;
        font-size: 20px;
        text-align: center;
        font-weight: bold;
    }}
    </style>
    """

    st.markdown(css, unsafe_allow_html=True)

# Apply background
add_bg_from_local("bg.jpg")

# Title
st.markdown('<p class="main-title">Spam vs Ham Classifier</p>', unsafe_allow_html=True)

# Load model
df = load_dataset()
model, vectorizer, metrics = train_model(df)

st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("Message Classification")

msg = st.text_area("Enter message:")

# Predict button
if st.button("Predict"):
    if msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        vec = vectorizer.transform([msg])
        result = model.predict(vec)[0]

        if result == 1:
            st.markdown('<div class="spam-box">SPAM</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ham-box">HAM</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
