import unittest
from unittest.mock import patch, PropertyMock
from tinygrad import UOp, dtypes
from tinygrad.device import Buffer
from tinygrad.runtime.support.hcq2 import fold_binary, fold_words
from tinygrad.uop.ops import Ops


class TestBulkPatch(unittest.TestCase):
  def check_patches(self, shadow):
    class Host:
      def __init__(self): self.data, self.writes = bytearray(64), 0
      def view(self, **kwargs): return self
      def __setitem__(self, index, data):
        self.data[index] = data
        self.writes += 1
    buf = Buffer('CPU', 64, dtypes.uint8, preallocate=True)
    if shadow: buf._hcq_patch_shadow = {}
    host, u = Host(), UOp.from_buffer(buf)
    initial = bytes(range(64))
    def words(offsets, values):
      fold_words(u, UOp.stack(*[UOp.const(x) for x in offsets]), UOp.stack(*[UOp.const(x, dtypes.uint32) for x in values]))
    with patch.object(Buffer, 'host', new_callable=PropertyMock, return_value=host):
      fold_binary(u, UOp(Ops.BINARY, arg=initial))
      words([0, 4], [100, 200])
      words([1, 5], [300, 400])
    expected = bytearray(initial)
    for off, value in ((0, 100), (16, 200), (4, 300), (20, 400)): expected[off:off+4] = value.to_bytes(4, 'little')
    self.assertEqual(host.data, expected)  # preserve holes and patches from preceding groups
    self.assertEqual(host.writes, 3 if shadow else 5)

  def test_bulk_preserves_gaps_and_prior_patches(self): self.check_patches(True)
  def test_unshadowed_buffer_keeps_individual_writes(self): self.check_patches(False)

  def test_fallback_updates_shadows_for_later_bulk_patch(self):
    buf = Buffer('CPU', 64, dtypes.uint8, preallocate=True)
    buf._hcq_patch_shadow = {}
    u = UOp.from_buffer(buf)
    fold_binary(u[:32], UOp(Ops.BINARY, arg=bytes(range(32))))
    fold_binary(u[32:], UOp(Ops.BINARY, arg=bytes(range(32, 64))))
    def words(offsets, values):
      fold_words(u, UOp.stack(*[UOp.const(x) for x in offsets]), UOp.stack(*[UOp.const(x, dtypes.uint32) for x in values]))
    words([2, 12], [900, 1000])  # spans two shadows, so uses individual writes
    words([1, 4], [1100, 1200])  # one bulk write with a hole covering the preceding patch at byte 8
    expected = bytearray(range(64))
    for at, value in ((8, 900), (48, 1000), (4, 1100), (16, 1200)): expected[at:at+4] = value.to_bytes(4, 'little')
    self.assertEqual(bytes(buf.as_memoryview()), expected)

  def test_overlapping_blob_invalidates_old_shadow(self):
    buf = Buffer('CPU', 64, dtypes.uint8, preallocate=True)
    buf._hcq_patch_shadow = {}
    u = UOp.from_buffer(buf)
    fold_binary(u, UOp(Ops.BINARY, arg=bytes(64)))
    fold_binary(u[8:16], UOp(Ops.BINARY, arg=bytes([99])*8))
    fold_words(u, UOp.stack(UOp.const(1), UOp.const(4)), UOp.stack(UOp.const(100, dtypes.uint32), UOp.const(200, dtypes.uint32)))
    self.assertEqual(bytes(buf.as_memoryview())[8:16], bytes([99])*8)


if __name__ == '__main__': unittest.main()
