import sys
sys.path.append('src')
import modem
import os
import unittest
import sounddevice as sd
import numpy as np
import time

class TestMain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.modem_a = modem.Modem()
        cls.modem_b = modem.Modem()

    def channelEffects(self, audio_data):
        return audio_data

    def testMain(self):
        payload = b"Hello World!"
        audio_data = self.modem_a.modulate(payload)
        received_audio_data = self.channelEffects(audio_data)
        received_payload = self.modem_b.demodulate(received_audio_data)
        self.assertEqual(received_payload, payload)

    def xtestModulation(self):
        payload = b"Hello World!"
        audio_data = self.modem_a.modulate(payload)
        sd.default.channels = 1
        sd.default.samplerate = 48000
        sd.play(audio_data, blocking=True)

    def xtestDemodulation(self):
        in_file = open("cq-modulation.bin", "rb")
        audio = in_file.read()
        in_file.close()
        data = self.modem_b.demodulate(audio)
        print(data)

if __name__ == '__main__':
    unittest.main()
