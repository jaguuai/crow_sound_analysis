# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:39:07 2025

@author: alice
"""
import librosa
import numpy as np
import librosa.display
import matplotlib.pyplot as plt

# WAV dosyasını yükle
file_path = 'C:/Users/alice/OneDrive/Masaüstü/CrowSoundAnaliysis/crow_sound.wav'  # Dosya yolunu buraya ekleyin
y, sr = librosa.load(file_path)

# 1. Temel frekans (pitch)
f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C1'), fmax=librosa.note_to_hz('C8'))
mean_freq = np.nanmean(f0)  # Ortalama frekans

# 2. RMS Enerji (sesin şiddeti)
rms = librosa.feature.rms(y=y)
mean_rms = np.mean(rms)

# 3. Ton stabilitesi (Pitch stability)
ton_stability = np.std(f0)  # Tonun ne kadar değişken olduğu

# 4. Gürültü oranı
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
mean_spectral_centroid = np.mean(spectral_centroid)

# 5. Zero crossing rate (sıfır noktasından geçiş sayısı)
zcr = librosa.feature.zero_crossing_rate(y=y)
mean_zcr = np.mean(zcr)

# Sesin sürekliliği (RMS)
temporal_features = librosa.feature.rms(y=y)

# Frekansa göre sınıflandırma yapalım
if mean_freq > 1000 and mean_rms > 0.02 and mean_zcr > 0.1:  # Yüksek frekanslar ve yüksek enerji genellikle alarm sesleridir
    classification = "Tehlike alarmı 🆘"
elif mean_freq > 300 and mean_spectral_centroid > 1000:  # Orta frekansta, daha keskin sesler iletişim çağrısı olabilir
    classification = "İletişim çağrısı 🗣️"
else:  # Düşük frekanslar, daha stabil ve yumuşak sesler şarkımsı/mırıldanma sesidir
    classification = "Şarkımsı veya mırıldanma sesi 🎶"

# Sonuçları Yazdır
print(f"Ses dosyasının sınıflandırması: {classification}")
print(f"Ortalama Frekans: {mean_freq:.2f} Hz")
print(f"Ortalama RMS Enerji: {mean_rms:.4f}")
print(f"Ton Stabilitesi (Std): {ton_stability:.2f}")
print(f"Spektral Merkez: {mean_spectral_centroid:.2f}")
print(f"Sıfır Noktası Geçiş Sayısı: {mean_zcr:.4f}")
