
import json
import sys
import traceback
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Labeling Tool",
    page_icon="🏷️",
    layout="centered"
)

st.title("🏷️ AI Labeling Tool")
st.caption(
    "Ubah laporan teks menjadi label: "
    "SAMPAH, LUBANG, KARATAN, RETAKAN"
)


# ============================================================
# CONFIG
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "indobert-labeler"
CONFIG_PATH = MODEL_PATH / "label_config.json"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True
    )

    model.eval()

    # Load custom config
    try:
        with open(
            CONFIG_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            config = json.load(f)

        labels = config["labels"]
        threshold_default = config.get("threshold", 0.5)
        max_length = config.get("max_length", 128)

    except FileNotFoundError:

        labels = [
            "SAMPAH",
            "LUBANG",
            "KARATAN",
            "RETAKAN"
        ]

        threshold_default = 0.5
        max_length = 128

    return (
        tokenizer,
        model,
        labels,
        threshold_default,
        max_length
    )


# ============================================================
# LOAD
# ============================================================

try:

    (
        tokenizer,
        model,
        LABELS,
        threshold_default,
        MAX_LENGTH
    ) = load_model()

except Exception as e:

    st.error(
        "Model IndoBERT belum ditemukan atau gagal dimuat."
    )

    st.write(f"**Python:** `{sys.executable}`")
    st.write(f"**Lokasi model:** `{MODEL_PATH}`")
    st.write(f"**Folder tersedia:** `{MODEL_PATH.is_dir()}`")
    st.exception(e)
    st.code(traceback.format_exc())

    st.info(
        "Jalankan Streamlit menggunakan interpreter yang sama "
        "dengan environment tempat `transformers` dan `torch` "
        "terpasang."
    )

    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

threshold = st.sidebar.slider(
    "Threshold",
    min_value=0.1,
    max_value=0.9,
    value=float(threshold_default),
    step=0.05,
    help="Label dipilih jika probabilitas >= threshold"
)

st.sidebar.markdown("---")

st.sidebar.write(
f"**Model:** `{MODEL_PATH.name}`"
)

st.sidebar.write(
    f"**Max tokens:** `{MAX_LENGTH}`"
)


# ============================================================
# PREDICTION
# ============================================================

def predict(texts):

    # Pastikan semuanya string
    texts = [
        str(text).lower().strip()
        for text in texts
    ]

    # Tokenisasi
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    # Inference
    with torch.no_grad():

        outputs = model(
            **inputs
        )

    # Multi-label -> sigmoid
    probabilities = torch.sigmoid(
        outputs.logits
    )

    return probabilities.cpu().numpy()


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    [
        "Satu teks",
        "Batch (CSV)"
    ]
)


# ============================================================
# TAB 1
# ============================================================

with tab1:

    text = st.text_area(
        "Masukkan laporan",
        "jalanan ini berlubang di area kampus",
        height=120,
        placeholder=(
            "Contoh: jalan depan gedung A "
            "banyak berlubang"
        )
    )

    if st.button(
        "Labeli",
        type="primary"
    ):

        if not text.strip():

            st.warning(
                "Masukkan laporan terlebih dahulu."
            )

        else:

            probs = predict([text])[0]

            # Pilih label berdasarkan threshold
            chosen = [
                label
                for label, probability
                in zip(LABELS, probs)
                if probability >= threshold
            ]

            # Result
            st.subheader("Hasil")

            if chosen:

                st.success(
                    "Label: "
                    + "  ".join(
                        f"`{label}`"
                        for label in chosen
                    )
                )

            else:

                st.warning(
                    "Tidak ada label yang cocok (NONE)"
                )

            # ==================================================
            # TABLE PROBABILITAS
            # ==================================================

            result_df = pd.DataFrame({
                "Label": LABELS,
                "Probabilitas": probs
            })

            result_df["Probabilitas"] = (
                result_df["Probabilitas"] * 100
            ).round(2)

            result_df["Status"] = result_df[
                "Probabilitas"
            ].apply(
                lambda x: (
                    "✅ Dipilih"
                    if x >= threshold * 100
                    else "❌ Tidak dipilih"
                )
            )

            st.dataframe(
                result_df,
                use_container_width=True,
                hide_index=True
            )

            # ==================================================
            # BAR CHART
            # ==================================================

            chart_df = pd.DataFrame(
                {
                    "probabilitas": probs
                },
                index=LABELS
            )

            st.bar_chart(
                chart_df
            )


# ============================================================
# TAB 2 - BATCH CSV
# ============================================================

with tab2:

    st.write(
        "Upload CSV dengan kolom `text`."
    )

    f = st.file_uploader(
        "CSV",
        type="csv"
    )

    if f:

        try:

            df = pd.read_csv(f)

        except Exception as e:

            st.error(
                f"Gagal membaca CSV: {e}"
            )

            st.stop()

        # Check column
        if "text" not in df.columns:

            st.error(
                "Kolom `text` tidak ditemukan."
            )

        else:

            with st.spinner(
                "Model IndoBERT sedang melakukan prediksi..."
            ):

                texts = (
                    df["text"]
                    .astype(str)
                    .tolist()
                )

                probs = predict(texts)

            # ==================================================
            # LABEL
            # ==================================================

            labels_result = []

            for row in probs:

                selected = [
                    label
                    for label, probability
                    in zip(LABELS, row)
                    if probability >= threshold
                ]

                if selected:
                    labels_result.append(
                        ",".join(selected)
                    )
                else:
                    labels_result.append(
                        "NONE"
                    )

            df["label"] = labels_result

            # ==================================================
            # PROBABILITY COLUMNS
            # ==================================================

            for i, label in enumerate(LABELS):

                df[
                    f"prob_{label.lower()}"
                ] = (
                    probs[:, i] * 100
                ).round(2)

            # ==================================================
            # DISPLAY
            # ==================================================

            st.success(
                f"{len(df)} data berhasil diproses."
            )

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            # ==================================================
            # DOWNLOAD
            # ==================================================

            csv_output = df.to_csv(
                index=False
            )

            st.download_button(
                "Download hasil",
                csv_output,
                "hasil_label.csv",
                "text/csv"
            )   