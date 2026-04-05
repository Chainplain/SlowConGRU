import numpy as np
import matplotlib.pyplot as plt


def generate_test_signal(duration=10.0, sampling_rate=50.0, target_freq=3.5):
    """
    Generate a test signal with a known frequency.
    """
    num_samples = int(duration * sampling_rate)
    time_array = np.linspace(0, duration, num_samples)

    signal_array = np.sin(2 * np.pi * target_freq * time_array) \
                   + 0.1 * np.random.randn(num_samples)

    return time_array, signal_array


def compute_fft(timestamps, signal_values):
    """
    Compute FFT from time and signal lists.

    Parameters:
    - timestamps: list or array of time values
    - signal_values: list or array of signal values

    Returns:
    - positive_freqs: frequency values (Hz)
    - positive_magnitude: magnitude spectrum
    """

    # Convert to numpy arrays
    t = np.array(timestamps)
    signal = np.array(signal_values)

    # Estimate sampling rate from timestamps
    dt = np.mean(np.diff(t))
    sampling_rate = 1.0 / dt

    N = len(signal)

    # Compute FFT
    fft_values = np.fft.fft(signal)
    fft_freqs = np.fft.fftfreq(N, d=dt)

    # Keep only positive frequencies
    positive_freqs = fft_freqs[:N // 2]
    positive_magnitude = np.abs(fft_values[:N // 2]) * 2 / N

    return positive_freqs, positive_magnitude


    # ==========================================
    # Example Usage
    # ==========================================

if __name__ == "__main__":
    # Generate test signal
    target_freq = 3.5
    t, signal = generate_test_signal(
        duration=10.0,
        sampling_rate=50.0,
        target_freq=target_freq
    )

    # Compute FFT
    freqs, magnitude = compute_fft(t.tolist(), signal.tolist())

    # Detect dominant frequency
    peak_index = np.argmax(magnitude)
    dominant_frequency = freqs[peak_index]

    print(f"Target frequency: {target_freq} Hz")
    print(f"Detected dominant frequency: {dominant_frequency:.3f} Hz")

    # Plot results
    plt.figure(figsize=(12, 5))

    # Time domain
    plt.subplot(1, 2, 1)
    plt.plot(t, signal)
    plt.title("Time Domain Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    # Frequency domain
    plt.subplot(1, 2, 2)
    plt.plot(freqs, magnitude)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()