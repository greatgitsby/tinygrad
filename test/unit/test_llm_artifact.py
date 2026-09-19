import json, pathlib, tempfile, unittest
from tinygrad import Device, Tensor, TinyJit, dtypes
from tinygrad.device import Buffer
from tinygrad.llm.artifact import load_artifact, save_artifact


class TestArtifact(unittest.TestCase):
  def test_aliases_and_deduplication(self):
    with tempfile.TemporaryDirectory() as root:
      path = pathlib.Path(root)/'artifact'
      base = Buffer(Device.DEFAULT, 64, dtypes.uint8, initial_value=bytes(range(64)))
      view = base.view(8, dtypes.uint8, 8).ensure_allocated()
      save_artifact((base, view, view), path)
      self.assertEqual(json.loads((path/'manifest.json').read_text())['buffers_bytes'], 64)
      (loaded, a, b), _ = load_artifact(path)
      self.assertIs(a, b)
      self.assertIs(a.base, loaded.base)
      self.assertEqual(bytes(a.ensure_allocated().as_memoryview()), bytes(range(8, 16)))
      with self.assertRaises(FileExistsError): save_artifact(base, path)

  def test_jit_roundtrip(self):
    with tempfile.TemporaryDirectory() as root:
      path = pathlib.Path(root)/'artifact'
      weights = Tensor([2.0, 3.0]).realize()
      run = TinyJit(lambda x: (x*weights+1).realize())
      for _ in range(2): run(Tensor([1.0, 2.0]).realize())
      save_artifact(run, path)
      restored, timings = load_artifact(path)
      self.assertEqual(restored(Tensor([4.0, 5.0]).realize()).tolist(), [9.0, 16.0])
      self.assertGreater(timings['bytes'], 0)
      with (path/'buffers.bin').open('r+b') as f: f.truncate(1)
      with self.assertRaisesRegex(ValueError, 'truncated'): load_artifact(path)

  def test_mutable_state_survives_roundtrip(self):
    with tempfile.TemporaryDirectory() as root:
      path = pathlib.Path(root)/'artifact'
      state = Tensor.zeros(2).contiguous().realize()
      def step(x): return state.assign(state+x).realize()
      run = TinyJit(step)
      for _ in range(2): run(Tensor([1.0, 2.0]).realize())
      expected = state.tolist()
      save_artifact(run, path)
      restored, _ = load_artifact(path)
      self.assertEqual(restored(Tensor([3.0, 4.0]).realize()).tolist(), [expected[0]+3, expected[1]+4])
      self.assertEqual(restored(Tensor([5.0, 6.0]).realize()).tolist(), [expected[0]+8, expected[1]+10])


if __name__ == '__main__': unittest.main()
