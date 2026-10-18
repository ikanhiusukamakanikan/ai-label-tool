"""Generate a varied 10,000-row multi-label dataset for campus reports.

The negative examples intentionally contain misleading words such as
"berlubang", "retak", and "karat" in personal or unrelated contexts. They
must remain NONE, for example: "celanaku berlubang".

Run:
    python generate_dataset.py
"""

import random
from pathlib import Path

import pandas as pd


SEED = 20261014
TARGET_ROWS = 10_000
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = PROJECT_DIR / "dataset.csv"
LABELS = ["SAMPAH", "LUBANG", "KARATAN", "RETAKAN"]

LOCATIONS = [
    "jalan depan gedung A", "trotoar samping perpustakaan",
    "akses menuju gedung B", "parkiran motor", "lapangan basket",
    "jalan belakang kampus", "area gerbang utama", "depan pos satpam",
    "jalan menuju rektorat", "halaman gedung D", "area parkir dosen",
    "dekat halte kampus", "koridor gedung C", "lantai dua gedung C",
    "lorong fakultas teknik", "samping laboratorium", "dekat kantin",
    "sekitar asrama mahasiswa", "depan aula", "taman kampus",
    "jalan antara gedung E dan F", "selokan belakang gedung B",
    "toilet lantai satu", "dekat lapangan futsal",
]

REPORT_OPENERS = [
    "", "", "lapor", "mohon dicek", "permisi", "izin melapor",
    "tolong ditindaklanjuti", "mohon perhatian", "info",
]

REPORT_CLOSERS = [
    "", "", "", "mohon segera diperbaiki", "cukup mengganggu",
    "membahayakan pengguna", "sudah beberapa hari", "sejak minggu lalu",
    "tolong dicek hari ini", "terlihat semakin parah",
]

ISSUES = {
    "SAMPAH": [
        "tumpukan sampah plastik berserakan",
        "bungkus makanan dan botol bekas menumpuk",
        "tempat sampah penuh sampai meluap",
        "sampah daun bercampur plastik belum diangkut",
        "puntung rokok berserakan di sekitar area",
        "bau sampah menyengat dari tumpukan yang membusuk",
        "kantong sampah robek dan isinya berhamburan",
        "sampah kertas memenuhi lantai",
        "gelas plastik bekas bertebaran",
        "selokan tersumbat sampah rumah tangga",
        "sisa makanan dibuang sembarangan",
        "tumpukan kardus bekas menghalangi jalan",
        "tong sampah rusak dan sampah berserakan",
        "sampah belum diangkut sejak kegiatan kampus",
        "banyak sampah menutup saluran air",
    ],
    "LUBANG": [
        "aspal berlubang cukup dalam",
        "jalan memiliki lubang besar di tengah jalur",
        "paving block amblas dan membentuk lubang",
        "permukaan jalan bolong-bolong",
        "lubang di trotoar membahayakan pejalan kaki",
        "aspal jebol setelah hujan",
        "lubang tergenang air di jalur kendaraan",
        "tanah amblas sehingga muncul lubang",
        "jalan rusak dengan beberapa lubang",
        "lubang pada akses masuk membuat kendaraan terguncang",
        "permukaan trotoar ambles dan berlubang",
        "lubang jalan makin melebar",
        "bagian aspal terkelupas hingga berlubang",
        "lubang terbuka belum diberi pembatas",
        "banyak lubang kecil di jalur menuju gedung",
    ],
    "KARATAN": [
        "pagar besi dipenuhi karat",
        "tiang besi sudah berkarat dan keropos",
        "railing tangga berkarat parah",
        "keran air berkarat",
        "engsel pintu besi mulai berkarat",
        "bangku besi penuh bercak karat",
        "rangka kanopi berkarat",
        "gerbang besi mengalami korosi",
        "rak besi berkarat dan catnya mengelupas",
        "tiang lampu berkarat",
        "pintu besi berkarat sulit dibuka",
        "besi penyangga terlihat keropos karena karat",
        "pagar pembatas lapangan berkarat",
        "meja besi berkarat",
        "saluran air dari besi mengalami karat",
    ],
    "RETAKAN": [
        "dinding gedung retak memanjang",
        "tembok memiliki retakan besar",
        "lantai beton retak",
        "plafon retak dan sebagian mengelupas",
        "keramik lantai pecah dan retak",
        "tiang bangunan terlihat retak",
        "tembok pagar retak",
        "retakan pada dinding semakin melebar",
        "permukaan lantai pecah-pecah",
        "dinding ruangan retak sampai dekat kusen",
        "tembok penahan mengalami retakan",
        "retakan terlihat di sepanjang koridor",
        "lantai teras gedung retak",
        "cat dinding mengelupas karena retakan",
        "bagian beton di sekitar tangga retak",
    ],
}

# These are deliberately hard negatives. The words overlap with labels, but
# the subject is personal, academic, social, or unrelated to campus facilities.
HARD_NEGATIVES = [
    "celanaku berlubang dan harus dijahit",
    "kaus kaki saya bolong sebelah",
    "tas kuliahku sobek di bagian bawah",
    "sepatu saya retak karena sudah lama dipakai",
    "layar ponselku retak setelah jatuh",
    "kaca mata saya retak tetapi masih bisa dipakai",
    "casing laptop retak di sudutnya",
    "telur rebus ini retak saat dimasak",
    "suara gitar itu agak sumbang dan senarnya putus",
    "buku catatan saya terkena noda kopi",
    "tugas kelompok belum selesai",
    "jadwal kuliah berubah mendadak",
    "wifi pribadi saya sedang lambat",
    "saya lupa membawa kartu mahasiswa",
    "teman saya membahas celana yang berlubang",
    "video tentang jalan berlubang sedang ditonton",
    "gambar tembok retak itu ada di buku pelajaran",
    "contoh kalimat ini menyebut kata karat",
    "filmnya menampilkan pagar tua penuh karat",
    "mainan mobil ini punya roda yang retak",
    "cangkir keramik di rumah retak",
    "pipa air di rumah saya berkarat",
    "sepeda pribadi saya berkarat di garasi rumah",
    "panci lama di dapur berkarat",
    "saya sedang mencari jasa jahit untuk celana berlubang",
    "pesan makanan untuk makan siang",
    "besok ada rapat organisasi mahasiswa",
    "cuaca hari ini hujan deras",
    "saya sedang belajar untuk ujian",
    "tolong kirimkan materi kuliah",
    "kucing lewat di depan rumah",
    "saya kehilangan pulpen warna biru",
    "musik dari kamar sebelah terlalu keras",
    "film baru itu ceritanya menarik",
    "saya membeli botol minum baru",
    "motor saya perlu dicuci",
]

NEUTRAL_REPORTS = [
    "lampu koridor mati", "AC ruang kelas tidak dingin",
    "proyektor tidak menyala", "air keran tidak mengalir",
    "pintu ruangan sulit dibuka", "lift sedang tidak berfungsi",
    "parkiran penuh", "suara pembangunan terlalu bising",
    "jendela kotor", "kursi kelas kurang", "colokan listrik tidak berfungsi",
    "taman terlihat rapi", "ruangan terlalu panas", "kucing berkeliaran",
    "mahasiswa sedang berkumpul", "jadwal penggunaan aula berubah",
]

NEGATIVE_OPENERS = [
    "", "", "catatan pribadi:", "cerita saya:", "untuk konteks pribadi:",
    "bukan laporan kampus:", "di luar kampus:", "sekadar cerita:",
]

NEGATIVE_DETAILS = [
    "", " tadi pagi", " sejak kemarin", " setelah dipakai lama",
    " di rumah", " saat perjalanan pulang", " untuk keperluan pribadi",
    " menurut cerita teman", " dalam tugas kuliah", " di video yang saya tonton",
]

NEGATIVE_CLOSERS = [
    "", "", " bukan fasilitas kampus", " tidak terkait area kampus",
    " ini urusan pribadi", " bukan laporan kerusakan",
]

CONNECTORS = [
    " dan ", ", sementara ", "; selain itu ", " serta ",
    ". Di lokasi yang sama, ", " lalu ",
]


def clean_sentence(parts: list[str]) -> str:
    opener, body, location, closer = parts
    sentence = body
    if location:
        sentence = f"{body} di {location}"
    if opener:
        sentence = f"{opener}: {sentence}"
    if closer:
        sentence = f"{sentence}; {closer}"
    return sentence.strip().capitalize()


def make_positive(labels: list[str], rng: random.Random) -> str:
    issue_parts = [rng.choice(ISSUES[label]) for label in labels]
    body = issue_parts[0]
    for issue in issue_parts[1:]:
        body += rng.choice(CONNECTORS) + issue

    location = rng.choice(LOCATIONS)
    opener = rng.choice(REPORT_OPENERS)
    closer = rng.choice(REPORT_CLOSERS)
    text = clean_sentence([opener, body, location, closer])

    # A small amount of realistic variation, without changing the label.
    if rng.random() < 0.15:
        text = text.replace(" yang ", " yg ")
    if rng.random() < 0.08:
        text = text.lower()
    return text


def make_negative(rng: random.Random) -> str:
    if rng.random() < 0.72:
        text = rng.choice(HARD_NEGATIVES)
    else:
        text = rng.choice(NEUTRAL_REPORTS)
        if rng.random() < 0.6:
            text += rng.choice([" di rumah", " untuk tugas", " hari ini", " saja"])

    text += rng.choice(NEGATIVE_DETAILS)
    text += rng.choice(NEGATIVE_CLOSERS)
    opener = rng.choice(NEGATIVE_OPENERS)
    if opener:
        text = f"{opener} {text}"
    return text.capitalize().replace("  ", " ")


def build_dataset() -> pd.DataFrame:
    rng = random.Random(SEED)
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    # 2,500 negatives teach the classifier that issue words alone are not enough.
    target_negative = 2_500
    while len(rows) < target_negative:
        text = make_negative(rng)
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        rows.append({"text": text, **{label: 0 for label in LABELS}})

    # Balanced single-label reports: 5,000 rows.
    for label in LABELS:
        count = 1_250
        added = 0
        while added < count:
            text = make_positive([label], rng)
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            rows.append({"text": text, **{item: int(item == label) for item in LABELS}})
            added += 1

    # Realistic multi-label reports: 2,500 rows.
    combinations = [
        ["SAMPAH", "LUBANG"],
        ["SAMPAH", "KARATAN"],
        ["SAMPAH", "RETAKAN"],
        ["LUBANG", "KARATAN"],
        ["LUBANG", "RETAKAN"],
        ["KARATAN", "RETAKAN"],
        ["SAMPAH", "LUBANG", "RETAKAN"],
        ["SAMPAH", "KARATAN", "RETAKAN"],
    ]
    for index in range(2_500):
        labels = combinations[index % len(combinations)]
        while True:
            text = make_positive(labels, rng)
            key = text.casefold()
            if key not in seen:
                break
        seen.add(key)
        rows.append({"text": text, **{item: int(item in labels) for item in LABELS}})

    rng.shuffle(rows)
    frame = pd.DataFrame(rows)
    frame["label"] = frame[LABELS].apply(
        lambda row: ",".join(label for label in LABELS if row[label]) or "NONE",
        axis=1,
    )
    return frame[["text", *LABELS, "label"]]


if __name__ == "__main__":
    dataset = build_dataset()
    dataset.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset tersimpan di: {OUTPUT_PATH}")
    print(f"Jumlah baris: {len(dataset)}")
    print("\nDistribusi label:")
    print(dataset["label"].value_counts().to_string())
