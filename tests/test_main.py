import sys
sys.path.append('src')
import modem

import unittest

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

if __name__ == '__main__':
    unittest.main()
