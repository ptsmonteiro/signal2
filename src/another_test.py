import numpy as np
import matplotlib.pyplot as plt
import komm
import math
import sounddevice as sd
from scipy.signal import spectrogram

SAMPLE_RATE = 8000
CENTER_FREQ = 1500
SYMBOL_DURATION_FACTOR = 1
FSK_BANDWIDTH = 2400
FSK_CARRIERS = 16

def plot_signal(signal, sample_rate = SAMPLE_RATE):
    t = np.arange(0, signal.size / sample_rate, 1/SAMPLE_RATE)

    if signal.size < t.size:
         signal = np.append(signal, np.zeros(t.size - signal.size, dtype=signal.dtype))

    plt.plot(t, np.real(signal))
    plt.title('Plot of a Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def plot_spectrum(signal, sample_rate = SAMPLE_RATE):
    freq_signal = np.abs(np.fft.fft(signal))
    freq_signal = freq_signal[0:freq_signal.size//2]
    f = np.arange(0, sample_rate/2, sample_rate / 2 / freq_signal.size)
    plt.plot(f, freq_signal)
    plt.title('Signal Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def plot_spectrogram(signal):
    f, t, Sxx = spectrogram(signal, SAMPLE_RATE)
    plt.pcolormesh(t, f, Sxx, shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.show()

def play_signal(signal, sample_rate = SAMPLE_RATE):
    sd.play(signal, samplerate=sample_rate)
    sd.wait()

def test_qam4():
    fft_size = 16

    for bits in [[0,0], [0,1], [1,0],[1,1]]:
        cenas = np.zeros(fft_size, dtype=np.complex128)
        print(cenas)

        print("bits", bits)
        point = komm.QAModulation(4).modulate(bits)[0]
        print("Constellation point", point)
        cenas[2] = point
        print(cenas)

        signal = np.fft.ifft(cenas)
        plot_signal(signal)

def fade_in_out(signal, freq, cycles = 2):
        """Apply a fade in and a fade out on a signal
        """
        fade_duration = cycles / freq
        fade_samples = fade_duration * SAMPLE_RATE

        fade = np.arange(1, 0, -1/fade_samples)
        signal[-fade.size:] *= fade

        fade = np.arange(0, 1, 1/fade_samples)
        signal[0:fade.size] *= fade

        return signal

def mod_fsk(data: bytes, bandwidth = FSK_BANDWIDTH, data_carriers = FSK_CARRIERS):
    """Modulates data into a MFSK signal according to specified bandwidth and 
    number of data carriers.
    """
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    print(f"{bits.size} bits")

    if (bits.size % data_carriers) > 0:
        bits = np.append(bits, np.zeros(data_carriers - (bits.size % data_carriers), dtype=np.uint8))

    print(f"{bits.size} bits after padding")

    assert(bits.size % data_carriers == 0)

    delta_freq = bandwidth // data_carriers
    print("delta freq", delta_freq)

    base_freq = CENTER_FREQ - bandwidth//2 + delta_freq//2
    print("base freq", base_freq)

    bits_per_symbol = int(math.log2(data_carriers))

    symbol_duration = SYMBOL_DURATION_FACTOR / delta_freq
    print(f"Symbol duration: {symbol_duration}s")
    print(f"Symbol samples {SAMPLE_RATE / delta_freq}")

    signal = np.zeros(0, dtype=np.complex128)

    prev_symbol_signal = None

    for i in range(0, bits.size, bits_per_symbol):
        symbol_bits = bits[i:i+bits_per_symbol]
        
        # Go from n bits to bytes padding as necessary
        left_zero_padding = np.zeros(8 - symbol_bits.size, dtype=np.uint8)
        carrier_index = np.packbits(np.append(left_zero_padding, symbol_bits))[0]
        
        freq_signal = np.zeros(SAMPLE_RATE, dtype=np.complex128)
        freq = base_freq + carrier_index*delta_freq
        print("Encoded symbol freq", freq)

        if prev_symbol_signal is None:
            freq_signal[freq] = np.exp(1j * np.pi/2)
        else:
            # This aligns the phase new symbol signal with the phase from 
            # the previous symbol signal
            freq_signal[freq] = np.exp(1j * (np.pi/2 + phase_next_sample))

        # 1s signal
        symbol_signal = np.fft.ifft(freq_signal)

        # trim to 2x symbol duration
        symbol_signal = symbol_signal[:int(2 * symbol_duration * SAMPLE_RATE)]

        # normalize
        symbol_signal = symbol_signal / np.max(np.real(symbol_signal))
                    
        # trim to symbol duration
        samples_duration = SAMPLE_RATE * symbol_duration
        symbol_signal = symbol_signal[0:round(samples_duration)]

        # Apply fading in the beginning and end of the symbol signal
        #symbol_signal = fade_in_out(symbol_signal, freq)

        signal = np.append(signal, symbol_signal)

        prev_symbol_signal = symbol_signal
        phase_offset_next_sample = (2 * np.pi * freq) / SAMPLE_RATE
        phase_next_sample = np.angle(symbol_signal[-1]) + phase_offset_next_sample
    
    audio = np.real(signal) / (100 * np.max(np.real(signal)))
    final_duration = audio.size / SAMPLE_RATE
    speed = len(data) / final_duration
    print(f"Transmission: {final_duration} s, {len(data)} bytes, {speed} bytes/s")
    #plot_signal(signal)
    #plot_spectrum(signal)
    return audio

def demod_fsk(signal, bandwidth = FSK_BANDWIDTH, data_carriers = FSK_CARRIERS):
    delta_freq = bandwidth // data_carriers
    print("delta freq", delta_freq)

    base_freq = CENTER_FREQ - bandwidth//2 + delta_freq//2
    print("base freq", base_freq)

    bits_per_symbol = int(math.log2(data_carriers))

    symbol_duration = SYMBOL_DURATION_FACTOR / delta_freq
    print(f"Symbol duration: {symbol_duration}s")
    symbol_samples = round(SAMPLE_RATE * symbol_duration)
    print(f"Symbol samples {symbol_samples}")

    i = 0
    while i < signal.size:
        symbol_signal = signal[i:i+symbol_samples]
        symbol_signal = np.append(symbol_signal, np.zeros(SAMPLE_RATE - symbol_signal.size, dtype=symbol_signal.dtype))
        #plot_signal(symbol_signal)
        #plot_spectrum(symbol_signal)

        freq_signal = np.abs(np.fft.fft(symbol_signal))
        freq_signal = freq_signal[0:round(freq_signal.size//2)]
        fi = np.argmax(freq_signal)
        #freq = (fi * SAMPLE_RATE) / (freq_signal.size * 2)
        print(f"Decoded symbol freq {fi}hz")

        i += symbol_samples

audio = mod_fsk(b"Lorem ipsum dolor sit amet, consectetur adipiscing elit.")
#play_signal(audio)
#plot_signal(audio)
#plot_spectrogram(audio)
bytes = demod_fsk(audio)
