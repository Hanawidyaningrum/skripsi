# 1. Import Library
import os
import re
import csv
import uuid
import shutil
import tempfile
import threading
import time
import glob
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import io
import json
import urllib.request
import urllib.parse
import torch
import torch.nn as nn
# Menyediakan operasi matematika untuk model, 
# seperti menghitung cosine similarity dan normalisasi embedding.
import torch.nn.functional as F
import paho.mqtt.client as mqtt
from flask import Flask, request, render_template, jsonify, session, redirect, url_for, send_file
from functools import wraps
from pydub import AudioSegment
# Digunakan untuk membaca dan menyimpan file audio, seperti WAV.
import soundfile as sf
# Digunakan untuk perhitungan angka dan sinyal audio, 
# misalnya normalisasi serta penghitungan kekuatan suara.
import numpy as np
import noisereduce as nr

import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

# Mengambil kelas SpeechBrain yang digunakan untuk memuat model pengenalan pembicara, 
# termasuk model X-Vector.
from speechbrain.inference.speaker import EncoderClassifier

# 2. Membuat Aplikasi Flask
app = Flask(__name__)

# 3. Login Admin
app.secret_key = "ganti_dengan_kode_rahasia_kamu_sendiri"

ADMIN_USERNAME = "hana"
ADMIN_PASSWORD = "hana123"
GOOGLE_STT_API_KEY = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def sanitize_name(nama):
    nama = (nama or "").strip().replace(" ", "_")
    nama = re.sub(r"[^A-Za-z0-9_\-]", "", nama)
    if not nama:
        raise ValueError("Nama pengguna tidak valid setelah dibersihkan.")
    return nama

db_lock = threading.Lock()

# 4. Membuat Folder Dataset
BASE_DIR = r"D:/skripsi/dataset"
DIR_PEMILIK = os.path.join(BASE_DIR, "suara_pemilik")
DIR_PENGUJIAN = os.path.join(BASE_DIR, "riwayat_pengujian")

os.makedirs(DIR_PEMILIK, exist_ok=True)
os.makedirs(DIR_PENGUJIAN, exist_ok=True)

# 5. Konfigurasi MQTT
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "skripsi/hanawidya/pagar"

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start()

# 6. Inisialisasi Model X-Vector
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Memuat model X-Vector di device: {DEVICE} ...")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-xvect-voxceleb",
    savedir="model_xvector",
    run_opts={"device": DEVICE}
)

# 7. ThreadPoolExecutor
executor_pool = ThreadPoolExecutor(max_workers=4)

# 8. Database Speaker Embedding
MASTER_VOICES_FILE = "master_voices_manual.pt"
global_master_embeddings = {}  # multi user

# 9. Database Threshold Per Pengguna
DATABASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "database")
THRESHOLD_FILE = os.path.join(DATABASE_DIR, "user_thresholds.txt")
os.makedirs(DATABASE_DIR, exist_ok=True)

user_thresholds = {}         # nama_user -> threshold skor centroid (float)
user_sample_thresholds = {}  # nama_user -> threshold skor per-sampel (float)

# 10. Log Pengujian (sesuai output terminal, disimpan ke CSV)
LOG_FILE = os.path.join(DATABASE_DIR, "log_pengujian.csv")
LOG_HEADER = [
    "Waktu",
    "Pengguna Terdeteksi",
    "Status",
    "Cosine Similarity",
    "Skor Web (%)",
    "Teks Terucap",
    "Perintah Terkirim",
    "Waktu Proses (detik)",
]
log_lock = threading.Lock()

def catat_log_pengujian(data):
    """Tambah satu baris hasil pengujian ke CSV.
    Pemisah titik koma (;) supaya Excel regional Indonesia membaca kolom dengan benar."""
    try:
        with log_lock:
            file_baru = not os.path.exists(LOG_FILE)
            with open(LOG_FILE, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=LOG_HEADER, delimiter=";")
                if file_baru:
                    writer.writeheader()
                writer.writerow(data)
    except Exception as e:
        print(f"[LOG] Gagal menulis log pengujian: {e}")

# 11. Kolom file threshold
THRESHOLD_FILE_FIELDS = ["nama", "threshold", "sample_threshold", "raw_dcf_threshold"]

def load_user_thresholds():
    """Baca threshold (centroid & sampel) per user dari database/user_thresholds.txt.
    Dibaca dengan csv.DictReader supaya kompatibel baik file format lama
    (8 kolom) maupun format baru (4 kolom) - kolom yang tidak dipakai
    cukup diabaikan, tidak menyebabkan error."""
    result = {}
    if not os.path.exists(THRESHOLD_FILE):
        return result

    with open(THRESHOLD_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nama = (row.get("nama") or "").strip()
            if not nama:
                continue
            try:
                threshold = float(row["threshold"])
            except (KeyError, TypeError, ValueError):
                threshold = BATAS_LOLOS
            try:
                sample_threshold = float(row["sample_threshold"])
            except (KeyError, TypeError, ValueError):
                sample_threshold = MIN_SAMPLE_THRESHOLD
            result[nama] = {"threshold": threshold, "sample_threshold": sample_threshold}
    return result

def save_user_thresholds(thresholds_detail):
    """Simpan threshold per user: nama, threshold final, sample_threshold,
    dan raw_dcf_threshold (hasil minDCF sebelum dinaikkan oleh floor)."""
    os.makedirs(DATABASE_DIR, exist_ok=True)

    with open(THRESHOLD_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=THRESHOLD_FILE_FIELDS)
        writer.writeheader()
        for nama, info in thresholds_detail.items():
            raw_dcf = info.get("raw_dcf_threshold")
            writer.writerow({
                "nama": nama,
                "threshold": info.get("threshold"),
                "sample_threshold": info.get("sample_threshold", MIN_SAMPLE_THRESHOLD),
                "raw_dcf_threshold": raw_dcf if raw_dcf is not None else "",
            })
    print(f"[THRESHOLD] {len(thresholds_detail)} threshold pengguna disimpan ke {THRESHOLD_FILE}")

def apply_thresholds(thresholds_detail):
    global user_thresholds, user_sample_thresholds
    user_thresholds = {n: i["threshold"] for n, i in thresholds_detail.items()}
    user_sample_thresholds = {
        n: i.get("sample_threshold", MIN_SAMPLE_THRESHOLD) for n, i in thresholds_detail.items()
    }
    save_user_thresholds(thresholds_detail)

# 12. Menentukan Threshold
BATAS_LOLOS = 0.76

COST_FAR = 5
COST_FRR = 1

P_TARGET = 0.5
P_NON = 1 - P_TARGET

MIN_SAFE_THRESHOLD = 0.76
CANDIDATE_MARGIN = 0.03
MIN_CONSISTENT_SAMPLES = 2       
MIN_SAMPLE_THRESHOLD = 0.60
SAMPLE_GENUINE_PERCENTILE = 25   
ADAPTER_SEED = 42

def set_seed(seed=ADAPTER_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 13. Pre-emphasis sebelum ekstraksi X-Vector. Kalau diubah, enroll ulang /
USE_PREEMPHASIS = True

# 14. Maksimal Durasi Audio
MAX_AUDIO_SECONDS = 5.0

# 15 Batas Kualitas Audio Minimum Saat Enroll
MIN_ENROLL_DURATION_SECONDS = 0.5
MIN_ENROLL_RMS = 0.01
MAX_ENROLL_CLIPPED_RATIO = 0.01

# 16. Konfigurasi Adapter
EMBED_DIM = 512
ADAPTER_HIDDEN = 256
TRIPLET_MARGIN = 0.2
ADAPTER_LR = 1e-3
ADAPTER_MAX_EPOCHS = 80
ADAPTER_TRIPLETS_PER_EPOCH = 200
ADAPTER_PATIENCE = 15
ADAPTER_SAVE_PATH = "adapter_finetuned.pt"

# 17. Hard Negative Ratio Adaptif
def compute_hard_ratio(n_users):
    return float(min(0.8, 0.3 + 0.05 * n_users))

# 18. Class Adapter
class ResidualAdapter(nn.Module):
    def __init__(self, dim=EMBED_DIM, hidden=ADAPTER_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        out = x + self.net(x)
        return F.normalize(out, p=2, dim=-1)

# 19. Memuat Adapter Tersimpan
voice_adapter = None
if os.path.exists(ADAPTER_SAVE_PATH):
    voice_adapter = ResidualAdapter().to(DEVICE)
    voice_adapter.load_state_dict(torch.load(ADAPTER_SAVE_PATH, map_location=DEVICE))
    voice_adapter.eval()
    print("Adapter suara terlatih berhasil dimuat.")
else:
    print("Belum ada adapter terlatih - sistem memakai x-vector mentah.")

# 20. Memuat Database Embedding
if os.path.exists(MASTER_VOICES_FILE):
    global_master_embeddings = torch.load(MASTER_VOICES_FILE)
    print(f"Berhasil memuat memori suara untuk {len(global_master_embeddings)} user.")

# 21. Memuat Threshold Per Pengguna
_loaded_thresholds = load_user_thresholds()
user_thresholds = {n: v["threshold"] for n, v in _loaded_thresholds.items()}
user_sample_thresholds = {n: v["sample_threshold"] for n, v in _loaded_thresholds.items()}
print(f"[THRESHOLD] Memuat {len(user_thresholds)} threshold pengguna dari database.")

# 22. Fungsi Ekstraksi X-Vector
def extract_xvector(audio_path, apply_adapter=True, preemph=None):
    if preemph is None:
        preemph = USE_PREEMPHASIS

    wav, _ = sf.read(audio_path, dtype='float32')
    wav_normalisasi = wav / (np.max(np.abs(wav)) + 1e-6)

    if preemph:
        ALPHA = 0.97
        wav = np.append(wav_normalisasi[0], wav_normalisasi[1:] - ALPHA * wav_normalisasi[:-1])
    else:
        wav = wav_normalisasi

    signal = torch.tensor(wav, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    embeddings = classifier.encode_batch(signal)

    if apply_adapter and voice_adapter is not None:
        with torch.no_grad():
            shape = embeddings.shape
            flat = embeddings.reshape(-1, EMBED_DIM)
            flat_out = voice_adapter(flat)
            embeddings = flat_out.reshape(shape)

    return embeddings.cpu()

# 23. Fungsi Hitung Centroid
def compute_centroid(embedding_list):
    stacked = torch.stack([e.squeeze() for e in embedding_list])
    normed = torch.nn.functional.normalize(stacked, p=2, dim=1)
    return normed.mean(dim=0)

# 24. Fungsi Hitung Threshold Berbasis Bobot Risiko (minDCF)
def compute_mindcf_threshold(genuine_scores, impostor_scores, cost_far=COST_FAR, cost_frr=COST_FRR, n_points=500):
    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)

    lo = min(genuine_scores.min(), impostor_scores.min())
    hi = max(genuine_scores.max(), impostor_scores.max())
    thresholds = np.linspace(lo, hi, n_points)

    far_list = np.array([(impostor_scores >= th).mean() for th in thresholds])
    frr_list = np.array([(genuine_scores < th).mean() for th in thresholds])

    total_cost = cost_frr * P_TARGET * frr_list + cost_far * P_NON * far_list
    idx = np.argmin(total_cost)

    return float(thresholds[idx]), float(far_list[idx]), float(frr_list[idx])

# 25. Fungsi Hitung FAR/FRR Aktual di Threshold Tertentu
def compute_far_frr_at_threshold(genuine_scores, impostor_scores, threshold):
    genuine_scores = np.asarray(genuine_scores)
    impostor_scores = np.asarray(impostor_scores)
    far = float((impostor_scores >= threshold).mean())
    frr = float((genuine_scores < threshold).mean())
    return far, frr

# 26. Fungsi Hitung Threshold Semua Pengguna (centroid + level-sampel)
def compute_user_thresholds(db=None):
    """
    db: dict {nama_user: [embedding, ...]} opsional (default: global_master_embeddings).
    Dipakai untuk evaluasi open-set dengan database "minus satu user".
    """
    db = global_master_embeddings if db is None else db
    users = list(db.keys())
    thresholds_detail = {}

    if len(users) < 2:
        for u in users:
            thresholds_detail[u] = {
                "threshold": BATAS_LOLOS,
                "sample_threshold": MIN_SAMPLE_THRESHOLD,
                "metode": "global_default",
                "n_genuine": len(db.get(u, [])), "n_impostor": 0,
            }
        return thresholds_detail

    for target_user in users:
        target_embs = [e.squeeze().float() for e in db[target_user]]
        n = len(target_embs)

        genuine_scores = []
        for i in range(n):
            held_out = target_embs[i]
            rest = target_embs[:i] + target_embs[i + 1:]
            if not rest:
                continue
            centroid = compute_centroid(rest)
            genuine_scores.append(F.cosine_similarity(centroid, held_out, dim=0).item())

        full_centroid = compute_centroid(target_embs)
        impostor_scores = []
        n_impostor_user_lain = 0
        for other_user in users:
            if other_user == target_user:
                continue
            for e in db[other_user]:
                impostor_scores.append(F.cosine_similarity(full_centroid, e.squeeze().float(), dim=0).item())
                n_impostor_user_lain += 1

        if len(genuine_scores) < 2 or len(impostor_scores) < 2:
            thresholds_detail[target_user] = {
                "threshold": BATAS_LOLOS,
                "sample_threshold": MIN_SAMPLE_THRESHOLD,
                "metode": "global_default",
                "n_genuine": len(genuine_scores), "n_impostor": len(impostor_scores),
            }
            continue

        dcf_th, far_raw, frr_raw = compute_mindcf_threshold(genuine_scores, impostor_scores)
        dcf_th_final = max(dcf_th, MIN_SAFE_THRESHOLD)
        floor_terpakai = dcf_th_final > dcf_th

        far_actual, frr_actual = compute_far_frr_at_threshold(genuine_scores, impostor_scores, dcf_th_final)

        gen_sample_scores = [
            F.cosine_similarity(target_embs[i], target_embs[j], dim=0).item()
            for i in range(n) for j in range(i + 1, n)
        ]
        imp_sample_scores = [
            F.cosine_similarity(t, e.squeeze().float(), dim=0).item()
            for other_user in users if other_user != target_user
            for e in db[other_user]
            for t in target_embs
        ]

        sample_threshold = MIN_SAMPLE_THRESHOLD
        if len(gen_sample_scores) >= 2 and len(imp_sample_scores) >= 2:
            s_dcf, _, _ = compute_mindcf_threshold(gen_sample_scores, imp_sample_scores)
            s_cap = float(np.percentile(gen_sample_scores, SAMPLE_GENUINE_PERCENTILE))
            sample_threshold = max(MIN_SAMPLE_THRESHOLD, min(s_dcf, s_cap))

        thresholds_detail[target_user] = {
            "threshold": round(dcf_th_final, 4),
            "sample_threshold": round(sample_threshold, 4),
            "raw_dcf_threshold": round(dcf_th, 4),
            "far_actual": round(far_actual, 4),
            "frr_actual": round(frr_actual, 4),
            "metode": (
                f"minDCF(cost_far={COST_FAR},cost_frr={COST_FRR})"
                + (f" + floor_keras({MIN_SAFE_THRESHOLD})" if floor_terpakai else "")
            ),
            "n_genuine": len(genuine_scores),
            "n_impostor": len(impostor_scores),
            "n_impostor_user_lain": n_impostor_user_lain,
        }

    return thresholds_detail

# 27. Fungsi Keputusan Identitas (Dipakai Bersama)
def decide_identity(ranking, user_thresholds_local, sample_scores_by_user=None,
                     sample_thresholds_local=None):
    """
    1. Skor centroid harus melewati threshold pengguna.
    2. Kandidat pertama harus unggul dari runner-up dengan margin minimum.
    3. Minimal beberapa sampel enrollment harus konsisten dengan suara pengujian.
    """
    if sample_thresholds_local is None:
        sample_thresholds_local = user_sample_thresholds

    kandidat_terdekat, best_score = ranking[0]
    batas_kandidat = user_thresholds_local.get(kandidat_terdekat, BATAS_LOLOS)

    runner_up_nama = None
    runner_up_score = None
    margin_aktual = None

    if len(ranking) >= 2:
        runner_up_nama, runner_up_score = ranking[1]
        margin_aktual = best_score - runner_up_score

    margin_cukup = True if margin_aktual is None else margin_aktual >= CANDIDATE_MARGIN

    consistent_count = 0
    # Kalau kandidat tidak punya sampel enrollment sama sekali, jangan
    # otomatis dianggap gagal konsistensi - anggap syarat ini terpenuhi,
    # supaya keputusan tetap ditentukan oleh threshold & margin.
    sample_consistency_ok = True
    sample_cutoff = sample_thresholds_local.get(kandidat_terdekat, MIN_SAMPLE_THRESHOLD)

    if sample_scores_by_user is not None:
        kandidat_sample_scores = sample_scores_by_user.get(kandidat_terdekat, [])
        if kandidat_sample_scores:
            required = min(MIN_CONSISTENT_SAMPLES, len(kandidat_sample_scores))
            consistent_count = sum(score >= sample_cutoff for score in kandidat_sample_scores)
            sample_consistency_ok = consistent_count >= required

    lolos_threshold = best_score >= batas_kandidat and margin_cukup and sample_consistency_ok

    runner_up_valid = (
        runner_up_score >= user_thresholds_local.get(runner_up_nama, BATAS_LOLOS)
        if runner_up_nama is not None else None
    )

    alasan_ditolak = None
    if not lolos_threshold:
        if best_score < batas_kandidat:
            alasan_ditolak = "di_bawah_threshold"
        elif not margin_cukup:
            alasan_ditolak = "margin_terlalu_tipis"
        else:
            alasan_ditolak = "suara_tidak_konsisten"

    return {
        "accepted": lolos_threshold,
        "predicted_user": kandidat_terdekat if lolos_threshold else None,
        "best_score": best_score,
        "kandidat_terdekat": kandidat_terdekat,
        "batas_kandidat": batas_kandidat,
        "runner_up_user": runner_up_nama,
        "runner_up_score": runner_up_score,
        "runner_up_valid": runner_up_valid,
        "margin_aktual": margin_aktual,
        "margin_cukup": margin_cukup,
        "consistent_sample_count": consistent_count,
        "sample_consistency_ok": sample_consistency_ok,
        "alasan_ditolak": alasan_ditolak,
    }

# 28. Helper: skor & keputusan untuk satu embedding uji terhadap satu database
def _score_and_decide(test_emb, db, thr_map, sthr_map):
    skor_per_user = {}
    sample_scores_by_user = {}
    for nama, embs in db.items():
        centroid = compute_centroid(embs)
        skor_per_user[nama] = F.cosine_similarity(centroid, test_emb, dim=0).item()
        sample_scores_by_user[nama] = [F.cosine_similarity(e, test_emb, dim=0).item() for e in embs]
    ranking = sorted(skor_per_user.items(), key=lambda kv: kv[1], reverse=True)
    return decide_identity(ranking, thr_map, sample_scores_by_user=sample_scores_by_user,
                            sample_thresholds_local=sthr_map)

def _threshold_maps(detail):
    thr = {n: i["threshold"] for n, i in detail.items()}
    sthr = {n: i.get("sample_threshold", MIN_SAMPLE_THRESHOLD) for n, i in detail.items()}
    return thr, sthr

# 29. Evaluasi FRR (pemilik ditolak) & FAR open-set (orang asing lolos)
def evaluate_closed_set_identification():
    """
    FRR: leave-one-out per sampel milik user terdaftar (apakah tetap dikenali).
    FAR (open-set): satu user dikeluarkan total dari database, lalu sampelnya
    dites seolah orang asing. Threshold dihitung ulang tiap kali tanpa data
    yang sedang diuji, supaya tidak ada data leakage.
    """
    db_full = {
        u: [e.squeeze().float() for e in embs]
        for u, embs in global_master_embeddings.items()
    }
    users = list(db_full.keys())
    if len(users) < 3:
        return {
            "status": "error",
            "msg": ("Minimal butuh 3 user terdaftar untuk evaluasi ini (saat satu user "
                    "dianggap orang asing, sistem tetap harus punya >=2 user pembanding)."),
        }

    genuine_result = {"benar": 0, "salah_user_lain": 0, "ditolak": 0}
    alasan_ditolak_count = {}
    detail_salah = []

    for target_user in users:
        for i in range(len(db_full[target_user])):
            rest = db_full[target_user][:i] + db_full[target_user][i + 1:]
            if not rest:
                continue
            db_i = {**db_full, target_user: rest}
            thr_map, sthr_map = _threshold_maps(compute_user_thresholds(db_i))
            hasil = _score_and_decide(db_full[target_user][i], db_i, thr_map, sthr_map)

            if not hasil["accepted"]:
                genuine_result["ditolak"] += 1
                alasan_ditolak_count[hasil["alasan_ditolak"]] = (
                    alasan_ditolak_count.get(hasil["alasan_ditolak"], 0) + 1
                )
            elif hasil["predicted_user"] == target_user:
                genuine_result["benar"] += 1
            else:
                genuine_result["salah_user_lain"] += 1
                detail_salah.append({
                    "suara_asli": target_user,
                    "terdeteksi_sebagai": hasil["predicted_user"],
                    "skor": round(hasil["best_score"], 4),
                })

    impostor_result = {"lolos": 0, "ditolak": 0}
    detail_lolos = []

    for stranger_user in users:
        db_minus = {u: embs for u, embs in db_full.items() if u != stranger_user}
        thr_map, sthr_map = _threshold_maps(compute_user_thresholds(db_minus))

        for e in db_full[stranger_user]:
            hasil = _score_and_decide(e, db_minus, thr_map, sthr_map)
            if hasil["accepted"]:
                impostor_result["lolos"] += 1
                detail_lolos.append({
                    "orang_asing": stranger_user,
                    "dikira_sebagai": hasil["predicted_user"],
                    "skor": round(hasil["best_score"], 4),
                })
            else:
                impostor_result["ditolak"] += 1

    n_genuine = sum(genuine_result.values())
    n_impostor = sum(impostor_result.values())
    frr = (genuine_result["ditolak"] + genuine_result["salah_user_lain"]) / n_genuine if n_genuine else None
    far = impostor_result["lolos"] / n_impostor if n_impostor else None

    print("\n" + "=" * 60)
    print("EVALUASI FRR (pemilik ditolak) & FAR open-set (orang asing lolos)")
    print("=" * 60)
    print(f"Total sampel genuine diuji  : {n_genuine}")
    print(f"  Benar teridentifikasi     : {genuine_result['benar']}")
    print(f"  Salah jadi user lain      : {genuine_result['salah_user_lain']}")
    print(f"  Ditolak (FRR)             : {genuine_result['ditolak']}")
    print(f"  Alasan ditolak            : {alasan_ditolak_count}")
    print(f"FRR                         : {frr:.4f}" if frr is not None else "FRR: -")
    print(f"Total sampel impostor diuji : {n_impostor}")
    print(f"  Lolos (FAR, BAHAYA)       : {impostor_result['lolos']}")
    print(f"  Ditolak (aman)            : {impostor_result['ditolak']}")
    print(f"FAR                         : {far:.4f}" if far is not None else "FAR: -")
    if detail_lolos:
        print("Detail orang asing yang lolos:")
        for d in detail_lolos:
            print(f"  {d['orang_asing']} -> dikira {d['dikira_sebagai']} (skor={d['skor']})")
    print("=" * 60 + "\n")

    return {
        "status": "success",
        "FRR": round(frr, 4) if frr is not None else None,
        "FAR": round(far, 4) if far is not None else None,
        "genuine": genuine_result,
        "alasan_pemilik_ditolak": alasan_ditolak_count,
        "detail_salah_identifikasi": detail_salah,
        "impostor": impostor_result,
        "detail_orang_asing_lolos": detail_lolos,
    }

# 30. Membangun Dataset Embedding
def build_embedding_dataset(label_names):
    embeddings, labels = [], []
    for label_idx, nama in enumerate(label_names):
        user_dir = os.path.join(DIR_PEMILIK, nama)
        wav_files = sorted(glob.glob(os.path.join(user_dir, "*.wav")))
        for wav_path in wav_files:
            emb = extract_xvector(wav_path, apply_adapter=False)
            embeddings.append(emb.squeeze().cpu())
            labels.append(label_idx)
    return torch.stack(embeddings), torch.tensor(labels)

# 31. Split Data Train dan Validasi
def split_train_val(embeddings, labels, val_ratio=0.2, min_for_split=5):
    """User dengan sampel < min_for_split dimasukkan seluruhnya ke train,
    supaya val_acc tidak dihitung dari 1 sampel per user."""
    train_idx, val_idx = [], []
    labels_np = labels.numpy()

    for label_val in np.unique(labels_np):
        idx = np.where(labels_np == label_val)[0]
        np.random.shuffle(idx)
        if len(idx) < min_for_split:
            train_idx.extend(idx)
            continue
        n_val = max(1, int(len(idx) * val_ratio))
        val_idx.extend(idx[:n_val])
        train_idx.extend(idx[n_val:])

    if not val_idx:
        val_idx = list(train_idx)

    return (
        embeddings[train_idx], labels[train_idx],
        embeddings[val_idx], labels[val_idx],
    )

def _labels_with_min_count(labels_np, min_count=2):
    unique, counts = np.unique(labels_np, return_counts=True)
    return [u for u, c in zip(unique, counts) if c >= min_count]

# 32. Sampling Triplet
def sample_triplets(labels, n_triplets):
    labels_np = labels.numpy()
    unique_labels = np.unique(labels_np)
    pos_candidates = _labels_with_min_count(labels_np, 2) or list(unique_labels)

    triplets = []
    for _ in range(n_triplets):
        pos_label = random.choice(pos_candidates)
        neg_choices = [l for l in unique_labels if l != pos_label]
        if not neg_choices:
            continue
        neg_label = random.choice(neg_choices)

        pos_indices = np.where(labels_np == pos_label)[0]
        neg_indices = np.where(labels_np == neg_label)[0]

        replace_needed = len(pos_indices) < 2
        a_idx, p_idx = np.random.choice(pos_indices, size=2, replace=replace_needed)
        n_idx = np.random.choice(neg_indices)

        triplets.append((a_idx, p_idx, n_idx))

    return triplets

# 33. Sampling Triplet dengan Hard Negative Mining
def sample_triplets_hard(current_embeddings, labels, n_triplets, hard_ratio=0.5):
    labels_np = labels.numpy()
    unique_labels = np.unique(labels_np)
    pos_candidates = _labels_with_min_count(labels_np, 2) or list(unique_labels)

    triplets = []
    embeddings_detached = current_embeddings.detach()

    for _ in range(n_triplets):
        pos_label = random.choice(pos_candidates)
        neg_choices = [l for l in unique_labels if l != pos_label]
        if not neg_choices:
            continue
        neg_label = random.choice(neg_choices)

        pos_indices = np.where(labels_np == pos_label)[0]
        neg_indices = np.where(labels_np == neg_label)[0]

        replace_needed = len(pos_indices) < 2
        a_idx, p_idx = np.random.choice(pos_indices, size=2, replace=replace_needed)

        if random.random() < hard_ratio and len(neg_indices) > 1:
            anchor_emb = embeddings_detached[a_idx].unsqueeze(0)
            neg_embs = embeddings_detached[neg_indices]
            sims = F.cosine_similarity(anchor_emb, neg_embs, dim=1)
            hardest_local_idx = torch.argmax(sims).item()
            n_idx = neg_indices[hardest_local_idx]
        else:
            n_idx = np.random.choice(neg_indices)

        triplets.append((a_idx, p_idx, n_idx))

    return triplets

# 34. Akurasi Triplet (Validasi)
def triplet_accuracy(adapter, embeddings, labels, n_eval=200):
    adapter.eval()
    triplets = sample_triplets(labels, n_eval)
    if not triplets:
        adapter.train()
        return 0.0
    correct = 0
    with torch.no_grad():
        out = adapter(embeddings)
        for a, p, n in triplets:
            d_pos = 1 - F.cosine_similarity(out[a], out[p], dim=0)
            d_neg = 1 - F.cosine_similarity(out[a], out[n], dim=0)
            if d_pos < d_neg:
                correct += 1
    adapter.train()
    return correct / len(triplets)

# 35. Membangun Ulang Database Embedding
def rebuild_master_embeddings(label_names):
    global global_master_embeddings

    new_db = {}
    for nama in label_names:
        user_dir = os.path.join(DIR_PEMILIK, nama)
        wav_files = sorted(glob.glob(os.path.join(user_dir, "*.wav")))
        emb_list = [extract_xvector(wav_path) for wav_path in wav_files]
        if emb_list:
            new_db[nama] = emb_list

    global_master_embeddings = new_db
    torch.save(global_master_embeddings, MASTER_VOICES_FILE)
    print(f"[ADAPTER] Database referensi diperbarui untuk {len(new_db)} pengguna.")

# 36. Ringkasan Embedding Untuk Response API
def build_adapter_summary(label_names):
    summary = {}
    for nama in label_names:
        emb_list = global_master_embeddings.get(nama, [])
        if not emb_list:
            continue
        centroid = compute_centroid(emb_list)
        summary[nama] = {
            "jumlah_sampel": len(emb_list),
            "centroid_preview": [round(v, 4) for v in centroid[:5].tolist()],
            "centroid_mean": round(centroid.mean().item(), 4),
            "centroid_std": round(centroid.std().item(), 4),
        }
    return summary

# 37. Fungsi Simpan dan Bersihkan File
def simpan_dan_bersihkan(wav_path, raw_path, nama_file_baru):
    try:
        shutil.move(wav_path, nama_file_baru)
    except PermissionError:
        shutil.copy(wav_path, nama_file_baru)
    except FileNotFoundError:
        pass
    try:
        if os.path.exists(raw_path):
            os.remove(raw_path)
    except Exception:
        pass

def bersihkan_file_sementara(*paths):
    for p in paths:
        try:
            if p and os.path.exists(p):
                os.remove(p)
        except Exception:
            pass

# 38. Fungsi Konversi Audio
def process_audio_file(web_audio_path, output_wav_path):
    audio = AudioSegment.from_file(web_audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_wav_path, format="wav")

# 39. Fungsi Penghilang Noise (Noise Reduction)
def reduce_noise_audio(wav_path, strength=0.6):
    try:
        data, samplerate = sf.read(wav_path, dtype='float32')
        if data.size == 0:
            return

        rms_before = float(np.sqrt(np.mean(data ** 2)))
        cleaned = nr.reduce_noise(y=data, sr=samplerate, stationary=False,
                                   prop_decrease=strength, n_jobs=-1)
        rms_after = float(np.sqrt(np.mean(cleaned ** 2)))

        if rms_before > 1e-6 and (rms_after / rms_before) < 0.4:
            print(f"[NOISE][WARNING] Energi sinyal turun drastis setelah denoise: "
                  f"{wav_path} | rms_before={rms_before:.4f} rms_after={rms_after:.4f}")

        sf.write(wav_path, cleaned, samplerate)
    except Exception as e:
        print(f"[NOISE] Gagal menghilangkan noise: {e}")

# 40. Fungsi Potong Durasi Audio
def trim_audio_if_too_long(wav_path, max_seconds=MAX_AUDIO_SECONDS):
    try:
        data, samplerate = sf.read(wav_path, dtype='float32')
        durasi_awal = len(data) / samplerate

        peak = np.max(np.abs(data)) + 1e-6
        data_relatif = data / peak

        SILENCE_THRESHOLD_RELATIF = 0.03
        above_threshold = np.where(np.abs(data_relatif) > SILENCE_THRESHOLD_RELATIF)[0]

        if len(above_threshold) > 0:
            padding_samples = int(0.2 * samplerate)
            start_idx = max(0, above_threshold[0] - padding_samples)

            waktu_mulai = start_idx / samplerate
            if waktu_mulai > durasi_awal * 0.5:
                print(f"[TRIM][WARNING] Titik awal suara terdeteksi di {waktu_mulai:.2f}s "
                      f"dari total {durasi_awal:.2f}s ({wav_path}) - kemungkinan awal ucapan terpotong.")

            data = data[start_idx:]

        durasi = len(data) / samplerate
        if durasi > max_seconds:
            batas_sampel = int(max_seconds * samplerate)
            data = data[:batas_sampel]
            durasi = max_seconds

        if durasi < 0.5:
            print(f"[TRIM][WARNING] Durasi audio akhir cuma {durasi:.2f}s ({wav_path}) - "
                  f"terlalu pendek untuk ekstraksi X-Vector yang stabil.")

        sf.write(wav_path, data, samplerate)
        return durasi
    except Exception as e:
        print(f"[TRIM] Gagal memotong durasi audio: {e}")
        return None


# 41. Fungsi Validasi Kualitas Audio
def check_audio_quality(wav_path):
    try:
        data, samplerate = sf.read(wav_path, dtype='float32')
        if data.size == 0:
            return False, "File audio kosong setelah preprocessing."

        durasi = len(data) / samplerate
        if durasi < MIN_ENROLL_DURATION_SECONDS:
            return False, (
                f"Durasi audio cuma {durasi:.2f}s setelah preprocessing (minimal "
                f"{MIN_ENROLL_DURATION_SECONDS}s). Coba rekam ulang dengan suara lebih jelas."
            )

        rms = float(np.sqrt(np.mean(data ** 2)))
        if rms < MIN_ENROLL_RMS:
            return False, f"Sinyal suara terlalu lemah (rms={rms:.4f})."

        clipped_ratio = float(np.mean(np.abs(data) > 0.99))
        if clipped_ratio > MAX_ENROLL_CLIPPED_RATIO:
            return False, (
                f"Audio terdeteksi clipping ({clipped_ratio*100:.1f}% sampel mentok). "
                f"Coba rekam dengan volume lebih rendah."
            )

        return True, None
    except Exception as e:
        return False, f"Gagal memeriksa kualitas audio: {e}"

# 42. Pipeline Preprocessing Terpusat
def preprocess_audio_pipeline(raw_path, wav_path, max_seconds=MAX_AUDIO_SECONDS, noise_strength=0.6):
    process_audio_file(raw_path, wav_path)
    reduce_noise_audio(wav_path, strength=noise_strength)
    trim_audio_if_too_long(wav_path, max_seconds=max_seconds)

# 43. Fungsi Speech-to-Text
def perform_stt(wav_path, language="id-ID", timeout=8):
    try:
        data, samplerate = sf.read(wav_path, dtype='int16')
        buffer = io.BytesIO()
        sf.write(buffer, data, samplerate, format='FLAC')
        flac_bytes = buffer.getvalue()
    except Exception as e:
        print(f"[STT] Gagal encode audio ke FLAC: {e}")
        return None, "unclear"

    try:
        query = urllib.parse.urlencode({
            "client": "chromium",
            "lang": language,
            "key": GOOGLE_STT_API_KEY,
        })
        url = f"http://www.google.com/speech-api/v2/recognize?{query}"
        req = urllib.request.Request(
            url,
            data=flac_bytes,
            headers={"Content-Type": f"audio/x-flac; rate={samplerate}"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response_text = response.read().decode("utf-8")
    except Exception as e:
        print(f"[STT] Gagal terhubung ke server Google: {e}")
        return None, "request_error"

    for line in response_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            result_obj = json.loads(line)
        except ValueError:
            continue

        result_list = result_obj.get("result") or []
        if result_list:
            alternatives = result_list[0].get("alternative") or []
            if alternatives and alternatives[0].get("transcript"):
                return alternatives[0]["transcript"].lower(), None

    return None, "unclear"

# 44. Sinkronisasi Database Embedding
def sync_master_database():
    global global_master_embeddings

    if not os.path.exists(MASTER_VOICES_FILE):
        print("[SYNC] Database embedding belum ditemukan, lewati sinkronisasi.")
        return

    master_embeddings = torch.load(MASTER_VOICES_FILE)

    folder_users = set()
    if os.path.exists(DIR_PEMILIK):
        for folder in os.listdir(DIR_PEMILIK):
            folder_path = os.path.join(DIR_PEMILIK, folder)
            if os.path.isdir(folder_path):
                folder_users.add(folder)

    database_users = set(master_embeddings.keys())
    deleted_users = database_users - folder_users

    if deleted_users:
        for user in deleted_users:
            del master_embeddings[user]
        torch.save(master_embeddings, MASTER_VOICES_FILE)
        print(f"[SYNC] Database diperbarui. {len(deleted_users)} user dihapus dari database.")
    else:
        print("[SYNC] Database sudah sinkron dengan folder dataset.")

    global_master_embeddings = master_embeddings

# 45. Routing Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('trainer'))
        else:
            error = "Username atau password salah!"
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# 46. Routing Web
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/trainer')
@login_required
def trainer():
    return render_template('trainer.html')

# 47. API Pendaftaran Suara
@app.route('/api/enroll', methods=['POST'])
@login_required
def enroll_voice():
    if 'audio' not in request.files:
        return jsonify({"status": "error", "msg": "Tidak ada audio"})

    try:
        nama_user = sanitize_name(request.form.get('nama_user', ''))
    except ValueError:
        return jsonify({"status": "error", "msg": "Nama pengguna kosong atau tidak valid!"})

    command_raw = request.form.get('command', '').strip().lower()
    if command_raw == 'buka':
        kata_perintah = "buka_pagar"
    elif command_raw == 'tutup':
        kata_perintah = "tutup_pagar"
    else:
        kata_perintah = "tanpa_perintah"

    audio_file = request.files['audio']
    uid = uuid.uuid4().hex
    raw_path = os.path.join(tempfile.gettempdir(), f"raw_enroll_{uid}.webm")
    wav_path = os.path.join(tempfile.gettempdir(), f"enroll_{uid}.wav")

    try:
        audio_file.save(raw_path)
        preprocess_audio_pipeline(raw_path, wav_path)

        ok_quality, quality_reason = check_audio_quality(wav_path)
        if not ok_quality:
            print(f"[ENROLL][DITOLAK] Sample untuk '{nama_user}' ditolak: {quality_reason}")
            return jsonify({"status": "error", "msg": f"Sample ditolak: {quality_reason}"})

        new_embedding = extract_xvector(wav_path)
        waktu = datetime.now().strftime("%d%b_%H%M%S")

        with db_lock:
            if nama_user in global_master_embeddings:
                global_master_embeddings[nama_user].append(new_embedding)
                jumlah = len(global_master_embeddings[nama_user])
                status_msg = f"Berhasil menambah sampel ke-{jumlah} ({kata_perintah}) untuk {nama_user}."
            else:
                global_master_embeddings[nama_user] = [new_embedding]
                jumlah = 1
                status_msg = f"Berhasil membuat profil baru untuk {nama_user} ({kata_perintah})."

            torch.save(global_master_embeddings, MASTER_VOICES_FILE)

            user_dir = os.path.join(DIR_PEMILIK, nama_user)
            os.makedirs(user_dir, exist_ok=True)
            nama_file_baru = os.path.join(user_dir, f"{nama_user}_{kata_perintah}_{jumlah}_{waktu}.wav")
            simpan_dan_bersihkan(wav_path, raw_path, nama_file_baru)

            if len(global_master_embeddings) >= 2:
                apply_thresholds(compute_user_thresholds())

        print(f"[TRAINER] {status_msg}")
        return jsonify({"status": "success", "msg": status_msg})

    except Exception as e:
        print(f"[ENROLL] Gagal memproses enrollment: {e}")
        return jsonify({"status": "error", "msg": f"Gagal memproses audio: {e}"})
    finally:
        bersihkan_file_sementara(raw_path, wav_path)

# 48. API Latih Adapter
@app.route('/api/train_adapter', methods=['POST'])
@login_required
def train_adapter_endpoint():
    global voice_adapter

    try:
        set_seed(ADAPTER_SEED)

        label_names = sorted([
            d for d in os.listdir(DIR_PEMILIK)
            if os.path.isdir(os.path.join(DIR_PEMILIK, d))
        ])
        if len(label_names) < 2:
            return jsonify({
                "status": "error",
                "msg": "Minimal butuh 2 pengguna terdaftar untuk melatih adapter."
            })

        print(f"[ADAPTER] Membangun dataset embedding untuk: {label_names}")
        embeddings, labels = build_embedding_dataset(label_names)
        print(f"[ADAPTER] Total {len(embeddings)} embedding terkumpul.")

        train_emb, train_lbl, val_emb, val_lbl = split_train_val(embeddings, labels)
        train_emb, val_emb = train_emb.to(DEVICE), val_emb.to(DEVICE)

        new_adapter = ResidualAdapter().to(DEVICE)
        optimizer = torch.optim.Adam(new_adapter.parameters(), lr=ADAPTER_LR)
        loss_fn = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1 - F.cosine_similarity(x, y),
            margin=TRIPLET_MARGIN,
        )

        hard_ratio = compute_hard_ratio(len(label_names))
        print(f"[ADAPTER] hard_ratio dipakai: {hard_ratio:.2f} ({len(label_names)} user)")

        best_val_acc = 0.0
        best_state = None
        epochs_no_improve = 0
        last_epoch = 0

        for epoch in range(1, ADAPTER_MAX_EPOCHS + 1):
            last_epoch = epoch
            new_adapter.train()

            with torch.no_grad():
                mining_out = new_adapter(train_emb)
            triplets = sample_triplets_hard(mining_out, train_lbl, ADAPTER_TRIPLETS_PER_EPOCH, hard_ratio=hard_ratio)
            if not triplets:
                print("[ADAPTER] Tidak bisa membentuk triplet training - hentikan training.")
                break

            a_idx = torch.tensor([t[0] for t in triplets])
            p_idx = torch.tensor([t[1] for t in triplets])
            n_idx = torch.tensor([t[2] for t in triplets])

            optimizer.zero_grad()
            out = new_adapter(train_emb)
            loss = loss_fn(out[a_idx], out[p_idx], out[n_idx])
            loss.backward()
            optimizer.step()

            val_acc = triplet_accuracy(new_adapter, val_emb, val_lbl)
            print(f"[ADAPTER] Epoch {epoch:3d} | loss={loss.item():.4f} | val_acc={val_acc:.3f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.clone() for k, v in new_adapter.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= ADAPTER_PATIENCE:
                    print(f"[ADAPTER] Early stop di epoch {epoch}.")
                    break

        if best_state is None:
            return jsonify({"status": "error", "msg": "Training gagal menghasilkan adapter yang valid."})

        new_adapter.load_state_dict(best_state)
        new_adapter.eval()
        torch.save(new_adapter.state_dict(), ADAPTER_SAVE_PATH)

        with db_lock:
            voice_adapter = new_adapter
            rebuild_master_embeddings(label_names)
            embedding_summary = build_adapter_summary(label_names)
            thresholds_detail = compute_user_thresholds()
            apply_thresholds(thresholds_detail)

        print(f"[ADAPTER] Training selesai. best_val_acc={best_val_acc:.3f}, epochs={last_epoch}, "
              f"total_embeddings={len(embeddings)}.")

        return jsonify({
            "status": "success",
            "msg": "Adapter suara berhasil dilatih dan threshold per pengguna diperbarui.",
            "best_val_acc": round(best_val_acc, 3),
            "epochs_run": last_epoch,
            "hard_ratio_used": round(hard_ratio, 2),
            "total_embeddings": len(embeddings),
            "embedding_summary": embedding_summary,
            "threshold_summary": thresholds_detail,
        })

    except Exception as e:
        print(f"[ADAPTER] Gagal melatih adapter: {e}")
        return jsonify({"status": "error", "msg": f"Gagal melatih adapter: {e}"})

# 49. API Daftar Pengguna
@app.route('/api/users', methods=['GET'])
@login_required
def list_users():
    with db_lock:
        users = [
            {"nama_user": nama, "jumlah_sampel": len(list_sampel)}
            for nama, list_sampel in global_master_embeddings.items()
        ]
    users.sort(key=lambda u: u["nama_user"].lower())
    return jsonify({"status": "success", "users": users})

# 50. API Hapus Pengguna
@app.route('/api/delete-user', methods=['POST'])
@login_required
def delete_user():
    global global_master_embeddings

    try:
        nama_user = sanitize_name(request.form.get('nama_user', ''))
    except ValueError:
        return jsonify({"status": "error", "msg": "Nama pengguna kosong atau tidak valid!"})

    with db_lock:
        user_dir = os.path.join(DIR_PEMILIK, nama_user)
        folder_ditemukan = os.path.exists(user_dir)
        embedding_ditemukan = nama_user in global_master_embeddings

        if not folder_ditemukan and not embedding_ditemukan:
            return jsonify({"status": "error", "msg": f"Pengguna '{nama_user}' tidak ditemukan."})

        if folder_ditemukan:
            shutil.rmtree(user_dir)

        master_embeddings = torch.load(MASTER_VOICES_FILE) if os.path.exists(MASTER_VOICES_FILE) else {}

        if nama_user in master_embeddings:
            del master_embeddings[nama_user]
            torch.save(master_embeddings, MASTER_VOICES_FILE)

        global_master_embeddings = master_embeddings
        apply_thresholds(compute_user_thresholds())

    print(f"[DELETE] Pengguna '{nama_user}' berhasil dihapus sepenuhnya.")
    return jsonify({"status": "success", "msg": f"Pengguna '{nama_user}' berhasil dihapus."})

# 51. API Ubah Nama Pengguna
@app.route('/api/rename-user', methods=['POST'])
@login_required
def rename_user():
    global global_master_embeddings

    try:
        nama_lama = sanitize_name(request.form.get('nama_lama', ''))
        nama_baru = sanitize_name(request.form.get('nama_baru', ''))
    except ValueError:
        return jsonify({"status": "error", "msg": "Nama lama dan nama baru wajib diisi dan valid!"})

    if nama_lama == nama_baru:
        return jsonify({"status": "error", "msg": "Nama baru sama dengan nama lama."})

    with db_lock:
        old_dir = os.path.join(DIR_PEMILIK, nama_lama)
        new_dir = os.path.join(DIR_PEMILIK, nama_baru)

        folder_ditemukan = os.path.exists(old_dir)
        embedding_ditemukan = nama_lama in global_master_embeddings

        if not folder_ditemukan and not embedding_ditemukan:
            return jsonify({"status": "error", "msg": f"Pengguna '{nama_lama}' tidak ditemukan."})

        if os.path.exists(new_dir) or nama_baru in global_master_embeddings:
            return jsonify({"status": "error", "msg": f"Nama '{nama_baru}' sudah digunakan oleh pengguna lain."})

        if folder_ditemukan:
            os.rename(old_dir, new_dir)
            for fname in os.listdir(new_dir):
                if fname.startswith(nama_lama + "_"):
                    fname_baru = nama_baru + fname[len(nama_lama):]
                    os.rename(os.path.join(new_dir, fname), os.path.join(new_dir, fname_baru))

        master_embeddings = torch.load(MASTER_VOICES_FILE) if os.path.exists(MASTER_VOICES_FILE) else {}

        if nama_lama in master_embeddings:
            master_embeddings[nama_baru] = master_embeddings.pop(nama_lama)
            torch.save(master_embeddings, MASTER_VOICES_FILE)

        global_master_embeddings = master_embeddings

        if nama_lama in user_thresholds or len(global_master_embeddings) >= 2:
            apply_thresholds(compute_user_thresholds())

    print(f"[RENAME] Pengguna '{nama_lama}' berhasil diubah namanya menjadi '{nama_baru}'.")
    return jsonify({
        "status": "success",
        "msg": f"Nama pengguna berhasil diubah dari '{nama_lama}' menjadi '{nama_baru}'."
    })

# 52. API Verifikasi Suara & Perintah
@app.route('/api/command', methods=['POST'])
def process_command():
    if 'audio' not in request.files:
        return jsonify({"status": "error", "msg": "Tidak ada audio"})

    with db_lock:
        if not global_master_embeddings:
            return jsonify({"status": "error", "msg": "Belum ada satupun user terdaftar."})
        db_snapshot = {u: list(embs) for u, embs in global_master_embeddings.items()}
        thr_snapshot = dict(user_thresholds)
        sthr_snapshot = dict(user_sample_thresholds)

    audio_file = request.files['audio']
    uid = uuid.uuid4().hex
    raw_path = os.path.join(tempfile.gettempdir(), f"raw_cmd_{uid}.webm")
    wav_path = os.path.join(tempfile.gettempdir(), f"cmd_{uid}.wav")

    try:
        audio_file.save(raw_path)
        t0 = time.time()

        preprocess_audio_pipeline(raw_path, wav_path)

        ok_quality, quality_reason = check_audio_quality(wav_path)
        if not ok_quality:
            print(f"[COMMAND] Audio ditolak karena kualitas: {quality_reason}")
            return jsonify({"status": "error", "msg": f"Audio tidak valid: {quality_reason}"})

        future_embedding = executor_pool.submit(extract_xvector, wav_path)
        future_stt = executor_pool.submit(perform_stt, wav_path)

        test_embedding = future_embedding.result()
        stt_text, stt_error = future_stt.result()
        waktu_proses = time.time() - t0

        test_emb = test_embedding.squeeze().float()

        skor_per_user = {}
        sample_scores_by_user = {}
        for nama, list_sampel in db_snapshot.items():
            centroid = compute_centroid(list_sampel)
            similarity = torch.nn.functional.cosine_similarity(centroid, test_emb, dim=0)
            skor_per_user[nama] = similarity.item()
            sample_scores_by_user[nama] = [
                torch.nn.functional.cosine_similarity(e.squeeze().float(), test_emb, dim=0).item()
                for e in list_sampel
            ]

        ranking = sorted(skor_per_user.items(), key=lambda kv: kv[1], reverse=True)

        hasil_keputusan = decide_identity(
            ranking, thr_snapshot,
            sample_scores_by_user=sample_scores_by_user,
            sample_thresholds_local=sthr_snapshot,
        )
        kandidat_terdekat = hasil_keputusan["kandidat_terdekat"]
        best_score = hasil_keputusan["best_score"]
        batas_kandidat = hasil_keputusan["batas_kandidat"]
        margin_cukup = hasil_keputusan["margin_cukup"]
        lolos_threshold = hasil_keputusan["accepted"]

        def sample_consistency_display(nama):
            cutoff = sthr_snapshot.get(nama, MIN_SAMPLE_THRESHOLD)
            scores = sample_scores_by_user.get(nama, [])
            if not scores:
                return None
            return sum(s >= cutoff for s in scores)

        perbandingan_user = [
            {
                "rank": rank,
                "nama": nama,
                "skor": round(skor, 4),
                "threshold": round(thr_snapshot.get(nama, BATAS_LOLOS), 4),
                "lolos": skor >= thr_snapshot.get(nama, BATAS_LOLOS),
                "sample_consistency": sample_consistency_display(nama),
            }
            for rank, (nama, skor) in enumerate(ranking, start=1)
        ]

        score = best_score
        waktu = datetime.now().strftime("%d%b_%H%M%S")

        if lolos_threshold:
            persentase_visual = min(int(((score - batas_kandidat) / (1.0 - batas_kandidat)) * 14 + 85), 99)
        else:
            persentase_visual = max(0, min(int(score * 100), 84))

        command = ""
        text = None

        if not lolos_threshold:
            pengguna_label = "anomali"
            status_label = "Ditolak - Margin Terlalu Tipis" if (best_score >= batas_kandidat and not margin_cukup) else "Ditolak"
        else:
            pengguna_label = kandidat_terdekat

            if stt_error == "request_error":
                status_label = "Diterima - Gagal Jaringan STT"
            elif stt_text is None:
                status_label = "Diterima - Suara Tidak Jelas"
            else:
                text = stt_text
                kata = [w.strip(".,!?;:\"'()-") for w in text.split()]

                idx_pagar = [i for i, w in enumerate(kata) if w == "pagar"]
                idx_buka = [i for i, w in enumerate(kata) if w == "buka"]
                idx_tutup = [i for i, w in enumerate(kata) if w == "tutup"]
                ada_jangan = "jangan" in kata

                ada_pagar = len(idx_pagar) > 0
                ada_buka_pagar = any(b < p for b in idx_buka for p in idx_pagar)
                ada_tutup_pagar = any(t < p for t in idx_tutup for p in idx_pagar)

                if not ada_pagar:
                    status_label = "Diterima - Bukan Kata Kunci"
                elif ada_jangan:
                    # Frasa negasi ("jangan buka/tutup pagar") sengaja tidak
                    # dieksekusi, supaya kesalahan STT pada kata negasi tidak
                    # berakibat perintah terbalik yang berbahaya.
                    status_label = "Diterima - Frasa Negasi Diabaikan"
                elif ada_buka_pagar and not ada_tutup_pagar:
                    command = "BUKA"
                    status_label = "Diterima Buka"
                elif ada_tutup_pagar and not ada_buka_pagar:
                    command = "TUTUP"
                    status_label = "Diterima Tutup"
                else:
                    status_label = "Diterima - Frasa Tidak Valid"

        print("\n" + "=" * 60)
        print("HASIL IDENTIFIKASI SUARA")
        print("=" * 60)
        print(f"Pengguna Terdeteksi : {pengguna_label}")
        print(f"Status              : {status_label}")
        if lolos_threshold:
            print(f"Cosine Similarity   : {score:.4f}  (threshold {kandidat_terdekat}: {batas_kandidat:.4f})")
        else:
            print(f"Cosine Similarity   : {score:.4f}")
        print(f"Skor Web            : {persentase_visual}%")
        if text:
            print(f"Teks Terucap        : {text}")
        if command:
            print(f"Perintah Terkirim   : {command}")
        print(f"Waktu Proses        : {waktu_proses:.2f} detik")
        print("=" * 60 + "\n")

        # Catat hasil pengujian ke CSV (isi sama dengan output terminal di atas)
        catat_log_pengujian({
            "Waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Pengguna Terdeteksi": pengguna_label,
            "Status": status_label,
            "Cosine Similarity": round(score, 4),
            "Skor Web (%)": persentase_visual,
            "Teks Terucap": text or "",
            "Perintah Terkirim": command,
            "Waktu Proses (detik)": round(waktu_proses, 2),
        })

        if not lolos_threshold:
            nama_file_baru = os.path.join(DIR_PENGUJIAN, f"DITOLAK_{persentase_visual}_{waktu}.wav")
            simpan_dan_bersihkan(wav_path, raw_path, nama_file_baru)
            return jsonify({
                "status": "ditolak",
                "msg": f"Akses Ditolak! Suara tidak dikenali ({persentase_visual}%)",
                "perbandingan_user": perbandingan_user,
            })

        recognized_user = kandidat_terdekat

        if status_label == "Diterima - Gagal Jaringan STT":
            simpan_dan_bersihkan(wav_path, raw_path, os.path.join(DIR_PENGUJIAN, f"ERROR_JARINGAN_{waktu}.wav"))
            return jsonify({
                "status": "error",
                "msg": "Gagal terhubung ke layanan Speech-to-Text. Periksa koneksi internet server.",
                "perbandingan_user": perbandingan_user,
            })

        if status_label == "Diterima - Suara Tidak Jelas":
            simpan_dan_bersihkan(wav_path, raw_path, os.path.join(DIR_PENGUJIAN, f"ERROR_STT_{waktu}.wav"))
            return jsonify({
                "status": "error",
                "msg": "Suara dikenali, tapi kata-katanya kurang jelas.",
                "perbandingan_user": perbandingan_user,
            })

        if status_label == "Diterima - Bukan Kata Kunci":
            simpan_dan_bersihkan(wav_path, raw_path, os.path.join(DIR_PENGUJIAN, f"BUKAN_KATAKUNCI_{waktu}.wav"))
            return jsonify({
                "status": "error",
                "msg": "Suara terdeteksi tapi bukan kata kunci",
                "perbandingan_user": perbandingan_user,
            })

        if status_label == "Diterima - Frasa Negasi Diabaikan":
            simpan_dan_bersihkan(wav_path, raw_path, os.path.join(DIR_PENGUJIAN, f"NEGASI_{waktu}.wav"))
            return jsonify({
                "status": "error",
                "msg": "Kata 'pagar' terdeteksi bersama kata 'jangan' - perintah diabaikan demi keamanan.",
                "perbandingan_user": perbandingan_user,
            })

        if status_label == "Diterima - Frasa Tidak Valid":
            simpan_dan_bersihkan(wav_path, raw_path, os.path.join(DIR_PENGUJIAN, f"GAGALKATA_{waktu}.wav"))
            return jsonify({
                "status": "error",
                "msg": "Kata 'pagar' terdeteksi, tapi frasa 'buka pagar' / 'tutup pagar' tidak ditemukan",
                "perbandingan_user": perbandingan_user,
            })

        mqtt_client.publish(MQTT_TOPIC, command)
        simpan_dan_bersihkan(wav_path, raw_path, os.path.join(DIR_PENGUJIAN, f"SUKSES_{recognized_user}_{command}_{waktu}.wav"))

        return jsonify({
            "status": "success",
            "msg": f"Teks terdeteksi: '{text}'. Akses Diberikan untuk {recognized_user}! Mengirim perintah: {command}",
            "perbandingan_user": perbandingan_user,
        })

    except Exception as e:
        print(f"[COMMAND] Gagal memproses audio: {e}")
        return jsonify({"status": "error", "msg": f"Gagal memproses audio: {e}"})
    finally:
        bersihkan_file_sementara(raw_path, wav_path)

# 53. API Evaluasi FRR & FAR
@app.route('/api/evaluate-identification', methods=['GET'])
@login_required
def evaluate_identification_endpoint():
    with db_lock:
        hasil = evaluate_closed_set_identification()
    return jsonify(hasil)

# 54. Fungsi Cetak Ringkasan Threshold/FAR/FRR ke Terminal
def print_threshold_summary_table(thresholds_detail):
    print("\n" + "=" * 90)
    print("RINGKASAN THRESHOLD PER-USER (minDCF) / FAR / FRR")
    print(f"(floor: {MIN_SAFE_THRESHOLD}, cost_far={COST_FAR}, cost_frr={COST_FRR})")
    print("=" * 90)
    header = (f"{'Nama':<18}{'Threshold':>11}{'SampleThr':>11}{'RawDCF':>9}"
              f"{'FAR':>7}{'FRR':>7}{'nGen':>6}{'nImp':>6}")
    print(header)
    print("-" * 90)
    for nama, info in sorted(thresholds_detail.items()):
        threshold = info.get("threshold")
        sample_th = info.get("sample_threshold", MIN_SAMPLE_THRESHOLD)
        raw_dcf = info.get("raw_dcf_threshold", threshold)
        far = info.get("far_actual")
        frr = info.get("frr_actual")
        n_gen = info.get("n_genuine", 0)
        n_imp = info.get("n_impostor", 0)
        far_str = f"{far:.3f}" if far is not None else "-"
        frr_str = f"{frr:.3f}" if frr is not None else "-"
        print(f"{nama:<18}{threshold:>11.4f}{sample_th:>11.4f}{raw_dcf:>9.4f}"
              f"{far_str:>7}{frr_str:>7}{n_gen:>6}{n_imp:>6}")
    print("=" * 90)
    print("Kalau semua threshold sama dengan floor: RawDCF semua user masih di")
    print("bawah floor, biasanya karena sample enrollment (jumlah/variasi) belum")
    print("cukup buat sistem membedakan tiap orang. Tambah sample lalu cek lagi.")
    print("=" * 90 + "\n")

# 55. API Ringkasan Threshold Semua Pengguna
@app.route('/api/threshold-summary', methods=['GET'])
@login_required
def threshold_summary_endpoint():
    with db_lock:
        if len(global_master_embeddings) < 2:
            return jsonify({
                "status": "error",
                "msg": "Minimal butuh 2 pengguna terdaftar untuk menghitung threshold.",
            })
        thresholds_detail = compute_user_thresholds()

    print_threshold_summary_table(thresholds_detail)

    far_values = [v["far_actual"] for v in thresholds_detail.values() if v.get("far_actual") is not None]
    frr_values = [v["frr_actual"] for v in thresholds_detail.values() if v.get("frr_actual") is not None]
    rata_far = round(sum(far_values) / len(far_values), 4) if far_values else None
    rata_frr = round(sum(frr_values) / len(frr_values), 4) if frr_values else None

    return jsonify({
        "status": "success",
        "cost_far": COST_FAR,
        "cost_frr": COST_FRR,
        "min_safe_threshold": MIN_SAFE_THRESHOLD,
        "candidate_margin": CANDIDATE_MARGIN,
        "min_sample_threshold": MIN_SAMPLE_THRESHOLD,
        "rata_rata_far": rata_far,
        "rata_rata_frr": rata_frr,
        "detail_per_user": thresholds_detail,
    })

# 56. API Unduh Log Pengujian (CSV cadangan)
@app.route('/api/download-log', methods=['GET'])
@login_required
def download_log():
    if not os.path.exists(LOG_FILE):
        return jsonify({"status": "error", "msg": "Belum ada log pengujian."})
    return send_file(LOG_FILE, as_attachment=True, download_name="log_pengujian.csv")

# 57. Menjalankan Aplikasi
if __name__ == '__main__':
    sync_master_database()
    if len(global_master_embeddings) >= 2:
        apply_thresholds(compute_user_thresholds())
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)