import os
import shutil                  
import time                    
from datetime import datetime  
import torch
import speech_recognition as sr
import paho.mqtt.client as mqtt
from flask import Flask, request, render_template, jsonify
from pydub import AudioSegment
import soundfile as sf 
import numpy as np

import torchaudio
if not hasattr(torchaudio, 'list_audio_backends'):
    torchaudio.list_audio_backends = lambda: ['soundfile']

from speechbrain.inference.speaker import EncoderClassifier

app = Flask(__name__)

# ================= 1. BUAT FOLDER DATASET =================
BASE_DIR = r"D:/skripsi/dataset"
DIR_PEMILIK = os.path.join(BASE_DIR, "suara_pemilik")
DIR_PENGUJIAN = os.path.join(BASE_DIR, "riwayat_pengujian")

os.makedirs(DIR_PEMILIK, exist_ok=True)
os.makedirs(DIR_PENGUJIAN, exist_ok=True)

# ================= 2. PENGATURAN MQTT =================
MQTT_BROKER = "maqiatto.com" 
MQTT_PORT = 1883
MQTT_TOPIC = "hanawidyaningrum212@gmail.com/pagar"
MQTT_USER = "hanawidyaningrum212@gmail.com" 
MQTT_PASS = "1234567890" 

mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1)
mqtt_client.username_pw_set(MQTT_USER, MQTT_PASS)
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
mqtt_client.loop_start() 

# ================= 3. INISIALISASI AI =================
print("Memuat model X-Vector...")
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="model_xvector")
recognizer = sr.Recognizer()

# Menyimpan data suara dengan file terpisah agar bersih
MASTER_VOICES_FILE = "master_voices_manual.pt" 
global_master_embeddings = {} 

BATAS_LOLOS = 0.94 # Toleransi untuk pengujian/command

if os.path.exists(MASTER_VOICES_FILE):
    global_master_embeddings = torch.load(MASTER_VOICES_FILE)
    print(f"Berhasil memuat memori suara untuk {len(global_master_embeddings)} user.")

def extract_xvector(audio_path):
    wav, fs = sf.read(audio_path, dtype='float32')
    wav = wav / (np.max(np.abs(wav)) + 1e-6)
    wav = np.append(wav[0], wav[1:] - 0.97 * wav[:-1])
    signal = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    embeddings = classifier.encode_batch(signal)
    return embeddings

def simpan_dan_bersihkan(wav_path, raw_path, nama_file_baru):
    try:
        time.sleep(0.1)
        shutil.move(wav_path, nama_file_baru)
    except PermissionError:
        shutil.copy(wav_path, nama_file_baru)
    try:
        if os.path.exists(raw_path): os.remove(raw_path)
    except Exception: pass 

def process_audio_file(web_audio_path, output_wav_path):
    audio = AudioSegment.from_file(web_audio_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_wav_path, format="wav")

# ================= 4. ROUTING WEB =================

@app.route('/')
def index(): return render_template('index.html')

@app.route('/trainer')
def trainer(): return render_template('trainer.html')

# Gunakan rute /api/enroll sesuai dengan kodingan aslimu 
# (Pastikan di HTML javascriptnya mengarah ke /api/enroll juga)
@app.route('/api/enroll', methods=['POST'])
def enroll_voice():
    global global_master_embeddings
    
    if 'audio' not in request.files:
        return jsonify({"status": "error", "msg": "Tidak ada audio"})

    nama_user = request.form.get('nama_user', '').strip()
    if not nama_user:
        return jsonify({"status": "error", "msg": "Nama pengguna kosong!"})
    
    nama_user = nama_user.replace(" ", "_")

    audio_file = request.files['audio']
    raw_path, wav_path = "temp_raw.webm", "temp_enroll.wav"
    audio_file.save(raw_path)
    process_audio_file(raw_path, wav_path)
    
    # --- FITUR BARU: DETEKSI KATA UNTUK NAMA FILE ---
    kata_perintah = "tanpa_perintah"
    try:
        with sr.AudioFile(wav_path) as source:
            # Fitur pembuang bising (noise reduction)
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="id-ID").lower()
            
            if "jangan buka" in text: kata_perintah = "jangan_buka"
            elif "jangan tutup" in text: kata_perintah = "jangan_tutup"
            elif "buka" in text: kata_perintah = "buka"
            elif "tutup" in text: kata_perintah = "tutup"
            else:
                kata_perintah = text.replace(" ", "_") 
    except sr.UnknownValueError:
        kata_perintah = "suara_tidak_jelas"
    except Exception:
        kata_perintah = "error_sistem"
    # ------------------------------------------------

    new_embedding = extract_xvector(wav_path)
    waktu = datetime.now().strftime("%d%b_%H%M%S")
    
    # LOGIKA PENYIMPANAN
    if nama_user in global_master_embeddings:
        global_master_embeddings[nama_user].append(new_embedding)
        jumlah = len(global_master_embeddings[nama_user])
        status_msg = f"Berhasil menambah sampel ke-{jumlah} ({kata_perintah}) untuk {nama_user}."
    else:
        global_master_embeddings[nama_user] = [new_embedding]
        jumlah = 1
        status_msg = f"Berhasil membuat profil baru untuk {nama_user} ({kata_perintah})."

    # Simpan memori AI
    torch.save(global_master_embeddings, MASTER_VOICES_FILE)
    
    # BUAT FOLDER KHUSUS USER
    user_dir = os.path.join(DIR_PEMILIK, nama_user)
    os.makedirs(user_dir, exist_ok=True)
    
    # SIMPAN FILE FISIK DENGAN NAMA PERINTAH
    nama_file_baru = os.path.join(user_dir, f"{nama_user}_{kata_perintah}_{jumlah}_{waktu}.wav")
    simpan_dan_bersihkan(wav_path, raw_path, nama_file_baru)
    
    print(f"[TRAINER] {status_msg}")
    return jsonify({"status": "success", "msg": status_msg})

@app.route('/api/command', methods=['POST'])
def process_command():
    global global_master_embeddings
    
    if not global_master_embeddings: 
        return jsonify({"status": "error", "msg": "Belum ada satupun user terdaftar."})

    audio_file = request.files['audio']
    raw_path, wav_path = "temp_raw_cmd.webm", "temp_cmd.wav"
    audio_file.save(raw_path)
    process_audio_file(raw_path, wav_path)

    test_embedding = extract_xvector(wav_path)
    best_score = -1.0
    recognized_user = "Unknown"

    for nama, list_sampel in global_master_embeddings.items():
        for known_embedding in list_sampel:
            similarity = torch.nn.functional.cosine_similarity(known_embedding.squeeze(), test_embedding.squeeze(), dim=0)
            score_sementara = similarity.item()
            if score_sementara > best_score:
                best_score = score_sementara
                recognized_user = nama
            
    score = best_score
    waktu = datetime.now().strftime("%d%b_%H%M%S")
    
    if score >= BATAS_LOLOS:
        persentase_visual = min(int(((score - BATAS_LOLOS) / (1.0 - BATAS_LOLOS)) * 14 + 85), 99)
    else:
        persentase_visual = int((score ** 5) * 100)

    print(f"\n[COMMAND LOG] Terdeteksi sbg '{recognized_user}'. Skor Asli: {score:.4f} | Web: {persentase_visual}%")
    
    if score < BATAS_LOLOS: 
        nama_file_baru = os.path.join(DIR_PENGUJIAN, f"DITOLAK_{persentase_visual}_{waktu}.wav")
        simpan_dan_bersihkan(wav_path, raw_path, nama_file_baru)
        return jsonify({"status": "ditolak", "msg": f"Akses Ditolak! Suara tidak dikenali ({persentase_visual}%)"})

    try:
        with sr.AudioFile(wav_path) as source:
            # Fitur pembuang bising (noise reduction) ditambahkan di sini
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data, language="id-ID").lower()
            
            command = ""
            if "jangan buka" in text: command = "TUTUP"
            elif "jangan tutup" in text: command = "BUKA"
            elif "buka" in text: command = "BUKA"
            elif "tutup" in text: command = "TUTUP"
            else:
                simpan_dan_bersihkan(wav_path, raw_path, os.path.join(DIR_PENGUJIAN, f"GAGALKATA_{waktu}.wav"))
                return jsonify({"status": "sukses_verifikasi", "msg": f"Hai {recognized_user}, suara cocok tapi '{text}' bukan perintah."})

            mqtt_client.publish(MQTT_TOPIC, command)
            simpan_dan_bersihkan(wav_path, raw_path, os.path.join(DIR_PENGUJIAN, f"SUKSES_{recognized_user}_{command}_{waktu}.wav"))
            
            return jsonify({"status": "success", "msg": f"Akses Diberikan untuk {recognized_user}! Mengirim perintah: {command}"})

    except sr.UnknownValueError:
        simpan_dan_bersihkan(wav_path, raw_path, os.path.join(DIR_PENGUJIAN, f"ERROR_STT_{waktu}.wav"))
        return jsonify({"status": "error", "msg": "Suara dikenali, tapi kata-katanya kurang jelas."})

if __name__ == '__main__':
    # Ditambahkan host='0.0.0.0' agar bisa diakses dari HP via Cloudflare/WiFi
    app.run(host='0.0.0.0', port=5000, debug=True)