import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI Labeling Tool", page_icon="🏷️")
st.title("🏷️ AI Labeling Tool")
st.caption("Ubah laporan teks menjadi label: SAMPAH, LUBANG, KARATAN, RETAKAN")


@st.cache_resource
def load():
    return joblib.load("model.joblib")


try:
    bundle = load()
except FileNotFoundError:
    st.error("model.joblib belum ada. Jalankan `python generate_dataset.py` lalu `python train_model.py`.")
    st.stop()

model, LABELS = bundle["model"], bundle["labels"]
threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.5, 0.05,
                              help="Label dipilih jika probabilitas >= threshold")


def predict(texts):
    return model.predict_proba([t.lower() for t in texts])


tab1, tab2 = st.tabs(["Satu teks", "Batch (CSV)"])

with tab1:
    text = st.text_area("Masukkan laporan", "jalanan ini berlubang di area kampus", height=100)
    if st.button("Labeli", type="primary") and text.strip():
        probs = predict([text])[0]
        chosen = [l for l, p in zip(LABELS, probs) if p >= threshold]
        if chosen:
            st.success("Label: " + "  ".join(f"`{l}`" for l in chosen))
        else:
            st.warning("Tidak ada label yang cocok (NONE)")
        st.bar_chart(pd.DataFrame({"probabilitas": probs}, index=LABELS))

with tab2:
    st.write("Upload CSV dengan kolom `text`.")
    f = st.file_uploader("CSV", type="csv")
    if f:
        df = pd.read_csv(f)
        if "text" not in df.columns:
            st.error("Kolom `text` tidak ditemukan.")
        else:
            probs = predict(df["text"].astype(str).tolist())
            df["label"] = [",".join(l for l, p in zip(LABELS, r) if p >= threshold) or "NONE" for r in probs]
            st.dataframe(df)
            st.download_button("Download hasil", df.to_csv(index=False), "hasil_label.csv", "text/csv")