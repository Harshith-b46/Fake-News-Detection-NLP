import streamlit as st
import pickle

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(
    page_title="Fake News Detector",
    layout="wide",
    page_icon="📰"
)

# Sidebar
st.sidebar.title("📰 Fake News Detector")
page = st.sidebar.radio(
    "Navigation",
    ["Predict News", "Upload File", "Prediction History", "About"]
)

# Store history
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- PREDICT ----------------
if page == "Predict News":

    st.title("📰 Fake News Detection System")
    st.caption("Analyze news content using NLP based machine learning model")

    colA, colB = st.columns(2)

    with colA:
        title = st.text_input("News Title")

    with colB:
        st.write("")
        st.write("Paste full news for best results")

    news = st.text_area("News Content", height=200)

    analyze = st.button("🔍 Analyze News", use_container_width=True)

    if analyze:

        if title.strip() == "" and news.strip() == "":
            st.warning("Please enter news text")
        else:
            text = title + " " + news
            vector = vectorizer.transform([text])

            prediction = model.predict(vector)[0]
            prob = model.predict_proba(vector)[0]

            fake = prob[0] * 100
            real = prob[1] * 100
            confidence = max(real, fake)

            st.divider()

            col1, col2 = st.columns([3,1])

            if prediction == 1:
                result = "REAL"
                col1.success("✅ This appears to be REAL news")
            else:
                result = "FAKE"
                col1.error("🚨 This appears to be FAKE news")

            col2.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Prediction Breakdown")

            st.write("Real News Probability")
            st.progress(int(real))

            st.write("Fake News Probability")
            st.progress(int(fake))

            st.caption("Prediction generated using trained NLP model")

            # save history
            st.session_state.history.append({
                "text": text[:120],
                "result": result,
                "real": real,
                "fake": fake
            })

# ---------------- FILE UPLOAD ----------------
elif page == "Upload File":

    st.title("📄 Upload News File")

    file = st.file_uploader(
        "Upload .txt news file",
        type=["txt"]
    )

    if file:

        text = file.read().decode("utf-8")

        st.subheader("Preview")
        st.info(text[:800])

        if st.button("Analyze File", use_container_width=True):

            vector = vectorizer.transform([text])
            prediction = model.predict(vector)[0]
            prob = model.predict_proba(vector)[0]

            fake = prob[0] * 100
            real = prob[1] * 100
            confidence = max(real, fake)

            col1, col2 = st.columns([3,1])

            if prediction == 1:
                st.success("✅ REAL NEWS")
            else:
                st.error("🚨 FAKE NEWS")

            col2.metric("Confidence", f"{confidence:.2f}%")

            st.subheader("Prediction Breakdown")

            st.write("Real News Probability")
            st.progress(int(real))

            st.write("Fake News Probability")
            st.progress(int(fake))

# ---------------- HISTORY ----------------
elif page == "Prediction History":

    st.title("📊 Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet")
    else:
        for item in reversed(st.session_state.history):

            st.divider()

            col1, col2 = st.columns([4,1])

            col1.write(item["text"])

            if item["result"] == "REAL":
                col2.success("REAL")
            else:
                col2.error("FAKE")

            st.caption(
                f"Real: {item['real']:.2f}% | Fake: {item['fake']:.2f}%"
            )

# ---------------- ABOUT ----------------
else:

    st.title("About Fake News Detector")

    st.write("""
This application detects whether a news article is **Real or Fake** using 
Natural Language Processing and Machine Learning.

### Features
• Real-time news prediction  
• Confidence score  
• File upload testing  
• Prediction history  
• Clean interactive UI  

### How it Works
1. Input news text  
2. Text converted using TF-IDF  
3. ML model analyzes patterns  
4. Output shown as Real or Fake  

### Technologies
• Python  
• Scikit-learn  
• Streamlit  
• NLP (TF-IDF)
""")

    st.info("Fake News Detection using NLP")