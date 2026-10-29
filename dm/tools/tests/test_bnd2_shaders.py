"""Criterion shader streams: alignment bits, stream offsets and bounded inflation."""
import os
import struct
import sys
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'gfx/tools'))
with patch.dict(os.environ, DM_GAME='0100620012D6E000'):
    import shader_split as ss


def bundle(payload, compressed=True, kind=8):
    stored = zlib.compress(payload) if compressed else payload
    d = bytearray(0x80)
    struct.pack_into('<4s5I', d, 0, b'bnd2', 3, 1, 0, 1, 0x30)
    struct.pack_into('<4I', d, 0x18, 0x80, 0x80, 0x80, 0x80)
    struct.pack_into('<I', d, 0x28, 0x19 if compressed else 8)
    struct.pack_into('<I', d, 0x30 + 0x14, len(payload) | 0x40000000)
    struct.pack_into('<I', d, 0x30 + 0x24, len(stored))
    struct.pack_into('<I', d, 0x30 + 0x44, kind)
    return d + stored


class Bnd2Shaders(unittest.TestCase):
    def test_second_stream_with_alignment_bits(self):
        for compressed in (False, True):
            with self.subTest(compressed=compressed), patch.object(ss, 'glslc_shaders', return_value=['shader']) as scan:
                self.assertEqual(ss.bnd2_shaders(bundle(b'shader output', compressed), '/test'), ['shader'])
                scan.assert_called_once_with(b'shader output', '/test#bnd2:0:1', need=False)

    def test_non_shader_resource_is_not_inflated(self):
        d = bundle(b'not zlib', kind=4)
        d[-1] ^= 255
        self.assertEqual(ss.bnd2_shaders(d, '/test'), [])

    def test_truncated_table(self):
        with self.assertRaisesRegex(SystemExit, 'invalid bnd2 resource table'):
            ss.bnd2_shaders(bundle(b'x')[:0x60], '/test')

    def test_stream_outside_bundle(self):
        d = bundle(b'x')
        struct.pack_into('<I', d, 0x30 + 0x34, len(d))
        with self.assertRaisesRegex(SystemExit, 'outside bundle'):
            ss.bnd2_shaders(d, '/test')

    def test_inflation_cannot_exceed_declared_size(self):
        d = bundle(b'x' * 10000)
        struct.pack_into('<I', d, 0x30 + 0x14, 1)
        with self.assertRaisesRegex(SystemExit, 'wrong decompressed length'):
            ss.bnd2_shaders(d, '/test')


if __name__ == '__main__':
    unittest.main()
