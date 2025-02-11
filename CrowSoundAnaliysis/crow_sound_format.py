# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 12:45:51 2025

@author: alice
"""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import noisereduce as nr
from pydub import AudioSegment

# 1️⃣ MP3'ü WAV formatına çevir
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

# 2️⃣ Gürültü Temizleme ve Frekans Analizi
def process_audio(wav_path):
    # Ses dosyasını yükle
    y, sr = librosa.load(wav_path, sr=None)

    # Gürültü temizleme
    reduced_noise = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.8)

    # 3️⃣ Spectrogram Görselleştirme
    plt.figure(figsize=(12, 6))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(reduced_noise)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Gürültü Temizlenmiş Spectrogram')
    plt.show()

    # 4️⃣ Temel Frekans Analizi
    pitches, magnitudes = librosa.piptrack(y=reduced_noise, sr=sr)
    mean_pitch = np.mean(pitches[pitches > 0])  # 0 olmayanları al

    print(f"💡 Ortalama Frekans: {mean_pitch:.2f} Hz")

# Kullanım
mp3_file = "crow_sound.mp3"  # MP3 dosyanın adı
wav_file = "crow_sound.wav"  # Çevrilecek WAV dosyası

convert_mp3_to_wav(mp3_file, wav_file)  # MP3'ü WAV'e çevir
process_audio(wav_file)  # İşleme ve analiz
