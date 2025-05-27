import os
import numpy as np
import h5py
import soundfile as sf

# === Configuration ===
clean_dir = "validfiles"   # Folder with 10 clean speech .wav files
output_path = "validset/cv.ex"           # Output HDF5 file
rms = 1.0                                      # For normalization

# === Helper: Add white noise ===
def add_noise(clean, snr_db=0):
    rms_clean = np.sqrt(np.mean(clean**2))
    noise = np.random.randn(len(clean))
    rms_noise = np.sqrt(np.mean(noise**2))
    snr_linear = 10**(snr_db / 10)
    scale = rms_clean / (np.sqrt(snr_linear) * rms_noise)
    return clean + noise * scale

# === Create validation HDF5 file ===
writer = h5py.File(output_path, 'w')

for idx, file in enumerate(sorted(os.listdir(clean_dir))[:10]):
    if not file.endswith('.wav'):
        continue

    clean_path = os.path.join(clean_dir, file)
    sph, sr = sf.read(clean_path)

    # Make sure it's mono
    if sph.ndim > 1:
        sph = sph[:, 0]

    # Add synthetic noise
    mix = add_noise(sph, snr_db=0)

    # Normalize power
    c = rms * np.sqrt(mix.size / np.sum(mix**2))
    mix *= c
    sph *= c

    # Create group in HDF5
    grp = writer.create_group(str(idx))
    grp.create_dataset('mix', data=mix.astype(np.float32), shape=mix.shape, chunks=True)
    grp.create_dataset('sph', data=sph.astype(np.float32), shape=sph.shape, chunks=True)

    print(f"[{idx}/10] Added {file} to cv.ex")

writer.close()
print("✅ Validation set saved to:", output_path)
