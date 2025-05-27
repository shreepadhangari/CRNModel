import os
import h5py
import numpy as np
import librosa

def generate_white_noise(length):
    return np.random.normal(0, 1, length)

def add_noise(clean, noise, snr_db):
    noise = noise[:len(clean)]
    clean_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    snr_linear = 10 ** (snr_db / 10)
    scale = np.sqrt(clean_power / (snr_linear * noise_power))
    noisy = clean + scale * noise
    return noisy

# CONFIG
clean_dir = 'cleanaudio'    # e.g., TIMIT/TRAIN
output_dir = 'dataset'      # save .ex files here
snrs = [0, 5, 10, 15]
rms = 1.0
os.makedirs(output_dir, exist_ok=True)

# Get list of all clean WAV files
clean_files = [os.path.join(dp, f) for dp, _, fs in os.walk(clean_dir) for f in fs if f.endswith('.wav')]

for idx, clean_path in enumerate(clean_files):
    clean, sr = librosa.load(clean_path, sr=None)

    # Generate synthetic noise
    noise = generate_white_noise(len(clean))

    # Randomly choose SNR
    snr = np.random.choice(snrs)

    # Mix and normalize
    mix = add_noise(clean, noise, snr)
    scale = rms * np.sqrt(len(mix) / np.sum(mix**2))
    mix *= scale
    clean *= scale

    # Save as HDF5 `.ex` file
    filename = f"tr_{idx}.ex"
    filepath = os.path.join(output_dir, filename)
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('mix', data=mix.astype(np.float32), chunks=True)
        f.create_dataset('sph', data=clean.astype(np.float32), chunks=True)

    if idx % 50 == 0:
        print(f"[{idx}/{len(clean_files)}] Processed {filename}")
