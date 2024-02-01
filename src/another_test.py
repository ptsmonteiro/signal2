import numpy as np
import matplotlib.pyplot as plt
import komm
import math
import sounddevice as sd

SAMPLE_RATE = 48000
CENTER_FREQ = 1500

def plot_signal(signal, sample_rate = SAMPLE_RATE):
    t = np.arange(0, signal.size / sample_rate, 1/SAMPLE_RATE)

    plt.plot(t, np.real(signal))
    plt.title('Plot of a Signal')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def plot_spectrum(signal, sample_rate = SAMPLE_RATE):
    freq_signal = np.real(np.fft.fft(signal))
    freq_signal = freq_signal[0:freq_signal.size//2]
    f = np.arange(0, sample_rate/2, sample_rate / 2 / freq_signal.size)
    plt.plot(f, freq_signal)
    plt.title('Signal Spectrum')
    plt.xlabel('Frequency')
    plt.ylabel('Amplitude')
    plt.grid(True)
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

def test_fsk(data: bytes, bandwidth = 500, data_carriers = 8):
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    print("unpacked bits", bits)

    assert(bits.size % data_carriers == 0)

    base_freq = CENTER_FREQ - bandwidth//2
    print("base freq", base_freq)
    delta_freq = bandwidth // data_carriers
    print("delta freq", delta_freq)

    bits_per_symbol = int(math.log2(data_carriers))

    symbol_duration = 1 / delta_freq
    print(f"Symbol duration: {symbol_duration}s")
    print(f"Symbol samples {SAMPLE_RATE / delta_freq}")

    signal = np.zeros(0, dtype=np.complex128)

    prev_symbol_signal = None
    prev_symbol_freq = None

    for i in range(0, bits.size, bits_per_symbol):
        symbol_bits = bits[i:i+bits_per_symbol]
        #print("symbol bits", symbol_bits)
        
        # Go from n bits to bytes padding as necessary
        left_zero_padding = np.zeros(8 - symbol_bits.size, dtype=np.uint8)
        carrier_index = np.packbits(np.append(left_zero_padding, symbol_bits))[0]
        
        freq_signal = np.zeros(SAMPLE_RATE)
        freq = base_freq + carrier_index*delta_freq
        print("freq", freq)
        freq_signal[freq] = 1

        symbol_signal = np.fft.ifft(freq_signal)

        if prev_symbol_signal is not None:
            print(f"Symbol period {round(SAMPLE_RATE / freq, 3)} samples")

            prev_symbol_period_samples = SAMPLE_RATE / prev_symbol_freq
            print(f"Previous signal period {prev_symbol_period_samples} samples")
            phase_offset = (SAMPLE_RATE * symbol_duration % prev_symbol_period_samples) / prev_symbol_period_samples # phase as percentage
            print(f"Phase offset {round(phase_offset*100,1)}%")

            # what to discard on the current symbol from the beginning to align phase
            start_trim_samples = (((1 - phase_offset) / freq) * SAMPLE_RATE)
            print(f"Trimming {start_trim_samples} samples for a {phase_offset*100}% phase offset")
            #print("Symbol before", symbol_signal[:10])
            symbol_signal = symbol_signal[round(start_trim_samples):]
            #print("Symbol after", symbol_signal[:10])

        # trim to symbol duration
        samples_duration = SAMPLE_RATE * symbol_duration
        symbol_signal = symbol_signal[0:round(samples_duration)]
        print("final symbol samples", symbol_signal.size)
        #plot_signal(symbol_signal)

        signal = np.append(signal, symbol_signal)

        prev_symbol_signal = symbol_signal
        prev_symbol_freq = freq
        

    audio = np.real(signal) / np.max(np.real(signal))
    final_duration = audio.size / SAMPLE_RATE
    speed = len(data) / final_duration
    print(f"Final duration: {final_duration}s, {speed} bytes/s")
    plot_signal(signal)
    #plot_spectrum(signal)
    play_signal(audio)


test_fsk(b'Hello World!')