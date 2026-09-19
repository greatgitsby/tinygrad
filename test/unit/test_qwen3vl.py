import unittest

from tinygrad import Tensor, UOp
from tinygrad.llm.model import TransformerConfig
from tinygrad.llm.kernels.amd import Linear
from tinygrad.llm.qwen3vl import Qwen3VL, Qwen3VLRunner, materialize_weights


class TestQwen3VL(unittest.TestCase):
  def test_resident_linear(self):
    import numpy as np
    layer = Linear(32, 16, bias=True)
    layer.weight = layer.weight.half().realize()
    layer.bias = layer.bias.half().realize()
    layer.resident_fp16 = True
    x = Tensor.randn(4, 32).realize()
    expected = x.half().numpy().astype(np.float32) @ layer.weight.numpy().astype(np.float32).T + layer.bias.numpy()
    np.testing.assert_allclose(layer(x).numpy(), expected, atol=2e-3, rtol=2e-3)

  def model(self):
    Tensor.manual_seed(17)
    model = Qwen3VL(TransformerConfig(num_blocks=2, dim=16, hidden_dim=32, n_heads=2, n_kv_heads=1,
                                    norm_eps=1e-6, vocab_size=32, head_dim=8, rope_theta=10000, rope_dim=8,
                                    v_head_dim=8, max_context=32), (2, 1, 1))
    materialize_weights(model)
    return model

  def test_decode_positions_change_after_capture(self):
    model, reference = self.model(), self.model()
    start = UOp.variable('test_vl_start', 0, 31)
    position = UOp.variable('test_vl_position', 0, 31)
    temp = Tensor([0.0]).realize()
    # Inspect the entire cache as well as tokens: a frozen position can coincidentally
    # return the same argmax, but writes to the wrong cache slot after capture.
    for i in range(6):
      token = Tensor([[i+1]]).realize()
      actual = model.decode_jit(token, start.bind(i), position.bind(i), temp).tolist()
      expected = reference._decode(token, i, i, temp).tolist()
      self.assertEqual(actual, expected)
      for a, b in zip(model.blk, reference.blk):
        self.assertEqual(a.cache_kv.tolist(), b.cache_kv.tolist())

  def test_frame_graph_matches_eager_after_capture(self):
    class Vision:
      def __call__(self, x): return x, [x], (1, 2, 2)
    runner = Qwen3VLRunner(self.model(), Vision(), [1, 2, 3], (1, 2), (1, 2, 2), max_new_tokens=5)
    for _ in range(5):
      image = Tensor.randn(1, 16).realize()
      self.assertEqual(runner(image).tolist(), runner.forward(image).tolist())

  def test_constrained_frame_graph(self):
    class Vision:
      def __call__(self, x): return x, [x], (1, 2, 2)
    runner = Qwen3VLRunner(self.model(), Vision(), [1, 2, 3], (1, 2), (1, 2, 2), 1, allowed_tokens=[4, 9, 12, 20])
    hidden = Tensor.randn(1, 1, 16).realize()
    logits = runner.model.output(runner.model.output_norm(hidden))[:, -1].tolist()[0]
    self.assertEqual(runner.output.weight.shape, (4, 16))
    selected = runner.output(runner.model.output_norm(hidden))[:, -1].tolist()[0]
    for actual, token in zip(selected, [4, 9, 12, 20]): self.assertAlmostEqual(actual, logits[token], places=5)
    self.assertEqual(runner._greedy(hidden).tolist(), [[max([4, 9, 12, 20], key=lambda token: logits[token])]])
    for _ in range(5):
      image = Tensor.randn(1, 16).realize()
      result = runner(image).tolist()
      self.assertEqual(result, runner.forward(image).tolist())
      self.assertIn(result[0][0], [4, 9, 12, 20])


if __name__ == '__main__':
  unittest.main()
