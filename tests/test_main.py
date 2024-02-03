import sys
sys.path.append('src')
import modem
import os
import unittest
import sounddevice as sd
import numpy as np
import time
import codec2

class TestMain(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    def __testMode(self, mode):
        m = modem.Modem(mode)
        payload = b"Hello World!"
        audio_data = m.modulate(payload)
        received_payload = m.demodulate(audio_data)
        self.assertEqual(received_payload[:len(payload)], payload)

    def testModeDataC0(self):
        self.__testMode(codec2.FREEDV_MODE.datac0)

    def testModeDataC1(self):
        self.__testMode(codec2.FREEDV_MODE.datac1)

    def testModeDataC13(self):
        self.__testMode(codec2.FREEDV_MODE.datac13)

    def testModeDataC3(self):
        self.__testMode(codec2.FREEDV_MODE.datac3)

    def testModeDataC4(self):
        self.__testMode(codec2.FREEDV_MODE.datac4)

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
