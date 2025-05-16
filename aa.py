import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
import requests
from scipy.signal import square, butter, lfilter
from datetime import datetime, timedelta, timezone

# -----------------------------
# 1. Genel Ayarlar
# -----------------------------
duration = 60      # saniye
sr = 44100         # örnekleme hızı
t = np.linspace(0, duration, int(sr * duration), endpoint=False)

# -----------------------------
# 2. NOAA'dan Son 1 Saatlik X-Işını Verisini Çek ve Zarfı Oluştur
# -----------------------------
def fetch_last_hour_envelope():
    url = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
    data = requests.get(url).json()

    now = datetime.now(timezone.utc)  # timezone-aware UTC zamanı
    one_hour_ago = now - timedelta(hours=1)

    flux_vals = []
    times = []
    for e in data:
        if e['energy'] != "0.1-0.8nm":
            continue
        ts = datetime.fromisoformat(e['time_tag'].replace('Z','+00:00'))  # UTC-aware
        if one_hour_ago <= ts <= now:
            times.append(ts)
            flux_vals.append(float(e['flux']))

    sorted_pairs = sorted(zip(times, flux_vals))
    flux = np.array([f for _, f in sorted_pairs])

    flux = (flux - flux.min()) / (flux.max() - flux.min())

    envelope = np.interp(
        np.linspace(0, len(flux)-1, len(t)),
        np.arange(len(flux)),
        flux
    )
    return envelope

envelope = fetch_last_hour_envelope()

# -----------------------------
# 3. Sentez Fonksiyonları
# -----------------------------
def additive_synthesis(base_freq, harmonics, env, t, gain=0.8):
    sig = sum((1/i)*np.sin(2*np.pi*base_freq*i*t) for i in range(1, harmonics+1))
    sig *= env * gain
    return sig / np.max(np.abs(sig))

def lowpass_filter(data, cutoff, sr, order=5):
    nyq = 0.5 * sr
    b, a = butter(order, cutoff/nyq, btype='low')
    return lfilter(b, a, data)

def subtractive_synthesis(freq, env, t, sr, gain=0.8):
    raw = square(2*np.pi*freq*t)
    filt = lowpass_filter(raw, 500, sr)
    sig = filt * env * gain
    return sig / np.max(np.abs(sig))

def wavetable_synthesis(freq, env, t, sr, gain=0.8):
    sine_tab = np.sin(2*np.pi*np.linspace(0,1,sr,endpoint=False))
    saw_tab  = np.linspace(-1,1,sr,endpoint=False)
    phase_inc = freq / sr
    phase = (phase_inc * np.arange(len(t))) % 1
    idx = (phase * sr).astype(int)
    sig = (sine_tab[idx] + saw_tab[idx]) * (env * gain) / 2
    return sig / np.max(np.abs(sig))

# -----------------------------
# 4. Sesleri Üret ve Fade Uygula
# -----------------------------
additive   = additive_synthesis(220, 6, envelope, t)
subtractive= subtractive_synthesis(220, envelope, t, sr)
wavetable  = wavetable_synthesis(220, envelope, t, sr)

fade_len = int(0.05 * sr)
fade = np.linspace(0, 1, fade_len)
for arr in (additive, subtractive, wavetable):
    arr[:fade_len]   *= fade
    arr[-fade_len:]  *= fade[::-1]

# -----------------------------
# 5. Dosyalara Yaz
# -----------------------------
sf.write("additive_60s.wav",   additive, sr)
sf.write("subtractive_60s.wav", subtractive, sr)
sf.write("wavetable_60s.wav",   wavetable, sr)

# -----------------------------
# 6. Görselleştirme Fonksiyonu
# -----------------------------
def plot_wave_and_spectrogram(sig, sr, title):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f"{title} — Waveform")
    librosa.display.waveshow(sig, sr=sr)
    plt.subplot(1, 2, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(sig)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f"{title} — Spectrogram")
    plt.tight_layout()
    plt.show()

# -----------------------------
# 7. Hepsini Görselleştir
# -----------------------------
plot_wave_and_spectrogram(additive,    sr, "Additive (60s)")
plot_wave_and_spectrogram(subtractive, sr, "Subtractive (60s)")
plot_wave_and_spectrogram(wavetable,   sr, "Wavetable (60s)")
