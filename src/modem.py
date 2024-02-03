import codec2
import ctypes
import numpy
import structlog

class Modem():

    AUDIO_SAMPLE_RATE_RX = 48000
    AUDIO_SAMPLE_RATE_TX = 48000
    MODEM_SAMPLE_RATE = codec2.api.FREEDV_FS_8000
    AUDIO_FRAMES_PER_BUFFER_RX = 2400 * 2

    MODES = [
        codec2.FREEDV_MODE.datac13,
        codec2.FREEDV_MODE.datac4,
        codec2.FREEDV_MODE.datac0,
        codec2.FREEDV_MODE.datac3,
        codec2.FREEDV_MODE.datac1,
    ]

    TUNING_RANGE = 50

    DEFAULT_MODE = 1

    def __init__(self, mode = MODES[DEFAULT_MODE]) -> None:
        self.logger = structlog.get_logger('Modem')
        self.set_mode(mode)
        self.resampler = codec2.resampler()

    def get_mode(self) -> codec2.FREEDV_MODE:
        return self.mode
    
    def set_mode(self, mode: codec2.FREEDV_MODE):
        self.mode = mode
        #self.logger.info(f"Mode {mode.name}")
        self.c2instance = ctypes.cast(codec2.api.freedv_open(mode.value), ctypes.c_void_p)
        codec2.api.freedv_set_tuning_range(
            self.c2instance,
            ctypes.c_float(float(-self.TUNING_RANGE)),
            ctypes.c_float(float(self.TUNING_RANGE)),
        )

    def get_bytes_per_frame(self):
        bpf = int(codec2.api.freedv_get_bits_per_modem_frame(self.c2instance) / 8)
        return bpf

    def __modulate_preamble__(self) -> bytes:
        samples = codec2.api.freedv_get_n_tx_preamble_modem_samples(self.c2instance)
        audio = ctypes.create_string_buffer(samples * 2) # 16 bit audio samples
        codec2.api.freedv_rawdatapreambletx(self.c2instance, audio)
        return bytes(audio)

    def __modulate_postamble__(self) -> bytes:
        samples = codec2.api.freedv_get_n_tx_postamble_modem_samples(self.c2instance)
        audio = ctypes.create_string_buffer(samples * 2) # 16 bit audio samples
        codec2.api.freedv_rawdatapostambletx(self.c2instance, audio)
        return bytes(audio)
    
    def __frame_crc__(self, payload: bytes) -> bytes:
        crc = ctypes.c_ushort(codec2.api.freedv_gen_crc16(payload, len(payload)))
        crc = crc.value.to_bytes(2, byteorder="big")
        #self.logger.debug("CRC: ", crc=crc)
        return crc

    def modulate(self, payload: bytes) -> numpy.array:
        # Frame: <preamble><payload of bytes_per_frame><postamble>
        # Payload: <bytes_per_frame -2><crc 2 bytes>

        bytes_per_frame = self.get_bytes_per_frame()
        payload_bytes_per_frame = bytes_per_frame - 2

        buffer = bytearray(payload_bytes_per_frame)
        buffer[: len(payload)] = payload
        crc = self.__frame_crc__(bytes(buffer))
        buffer += crc

        # Modulate data
        payload_samples = codec2.api.freedv_get_n_tx_modem_samples(self.c2instance)
        payload_audio = ctypes.create_string_buffer(payload_samples * 2)

        c2buffer = (ctypes.c_ubyte * bytes_per_frame).from_buffer_copy(buffer)
        codec2.api.freedv_rawdatatx(self.c2instance, payload_audio, c2buffer)

        preamble = self.__modulate_preamble__()
        postamble = self.__modulate_postamble__()
        #self.logger.debug(f"Preamble {len(preamble)} bytes, payload_audio {len(payload_audio)} bytes, postamble {len(postamble)} bytes")
        #audio_out_bytes = bytes(8000*2) + preamble + payload_audio + postamble + bytes(8000*2)
        audio_out_bytes = preamble + payload_audio + postamble

        audio_out_numpy = numpy.frombuffer(audio_out_bytes, dtype=numpy.int16)
        audio48k = self.resampler.resample8_to_48(audio_out_numpy)
        return audio48k

    def demodulate(self, audio48k: numpy.array) -> bytes:

        audio8k = self.resampler.resample48_to_8(audio48k)

        bytes_per_frame = self.get_bytes_per_frame()
        bytes_out = ctypes.create_string_buffer(bytes_per_frame)

        offset = 0
        nin = codec2.api.freedv_nin(self.c2instance)
        while offset < audio8k.size:
            #self.logger.debug(f"Decoding offset {offset} with length {nin} ({offset+nin}/{audio8k.size})")
            #self.logger.debug("Decoding", audio = audio8k[offset:offset+nin])
            nbytes = codec2.api.freedv_rawdatarx(self.c2instance, 
                                                 bytes_out, 
                                                 audio8k[offset:offset+nin].ctypes)

            offset += nin

            # 1 trial
            # 2 sync
            # 3 trial sync
            # 6 decoded
            # 10 error decoding == NACK

            rx_status = codec2.api.freedv_get_rx_status(self.c2instance)
            self.logger.debug(
                        "Codec2 RX STATUS: ", mode=self.mode.name, rx_status=rx_status,
                        sync_flag=codec2.api.rx_sync_flags_to_text[rx_status])

            nin = codec2.api.freedv_nin(self.c2instance)
            payload = bytes(bytes_out)

            if rx_status == 6:
                self.logger.debug(f"Decoded {nbytes} bytes.", payload = payload)
            
        return payload
