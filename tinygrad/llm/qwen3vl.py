from __future__ import annotations

from pathlib import Path

from tinygrad import Tensor, TinyJit, UOp, dtypes, nn
from tinygrad.llm.gguf import gguf_load
from tinygrad.llm.kernels.amd import Linear
from tinygrad.llm.model import Transformer, TransformerConfig


class PatchEmbed:
  def __init__(self, dim:int, patch_size:int):
    self.weight0 = Tensor.empty(dim, 3, patch_size, patch_size)
    self.weight1 = Tensor.empty(dim, 3, patch_size, patch_size)
    self.bias = Tensor.empty(dim)
    self.patch_size = patch_size

  def __call__(self, image:Tensor) -> Tensor:
    # Image inputs are repeated across the two-frame temporal kernel, matching the reference image processor.
    x = image.conv2d(self.weight0 + self.weight1, self.bias, stride=self.patch_size)
    return x.permute(0, 2, 3, 1).reshape(-1, x.shape[1])


class VisionAttention:
  def __init__(self, dim:int, n_heads:int):
    self.n_heads, self.head_dim = n_heads, dim // n_heads
    self.attn_qkv = Linear(dim, dim * 3, bias=True)
    self.attn_out = Linear(dim, dim, bias=True)

  @staticmethod
  def _rotate_half(x:Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return (-x2).cat(x1, dim=-1)

  def __call__(self, x:Tensor, cos:Tensor, sin:Tensor) -> Tensor:
    n = x.shape[0]
    qkv = self.attn_qkv(x).reshape(n, 3, self.n_heads, self.head_dim)
    q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    q, k = q*cos + self._rotate_half(q)*sin, k*cos + self._rotate_half(k)*sin
    q, k, v = (z.transpose(0, 1).unsqueeze(0) for z in (q, k, v))
    out = q.scaled_dot_product_attention(k, v).transpose(1, 2).reshape(n, -1)
    return self.attn_out(out)


class VisionBlock:
  def __init__(self, dim:int, hidden_dim:int, n_heads:int, eps:float):
    self.ln1, self.ln2 = nn.LayerNorm(dim, eps), nn.LayerNorm(dim, eps)
    self.attn = VisionAttention(dim, n_heads)
    self.ffn_up = Linear(dim, hidden_dim, bias=True)
    self.ffn_down = Linear(hidden_dim, dim, bias=True)

  def __call__(self, x:Tensor, cos:Tensor, sin:Tensor) -> Tensor:
    x = x + self.attn(self.ln1(x), cos, sin)
    return x + self.ffn_down(self.ffn_up(self.ln2(x)).gelu())


class VisionMerger:
  def __init__(self, vision_dim:int, output_dim:int, eps:float, postshuffle_norm:bool):
    merged_dim = vision_dim * 4
    self.norm = nn.LayerNorm(merged_dim if postshuffle_norm else vision_dim, eps)
    self.fc1, self.fc2 = Linear(merged_dim, merged_dim, bias=True), Linear(merged_dim, output_dim, bias=True)
    self.postshuffle_norm = postshuffle_norm

  def __call__(self, x:Tensor) -> Tensor:
    x = self.norm(x.reshape(-1, x.shape[-1]*4) if self.postshuffle_norm else x).reshape(-1, x.shape[-1]*4)
    return self.fc2(self.fc1(x).gelu(approximate='none'))


class _VisionWeights:
  def __init__(self, dim:int, hidden_dim:int, output_dim:int, depth:int, n_heads:int, patch_size:int, eps:float):
    self.patch_embd = PatchEmbed(dim, patch_size)
    self.position_embd = nn.Embedding(48*48, dim)
    self.blk = [VisionBlock(dim, hidden_dim, n_heads, eps) for _ in range(depth)]
    self.deepstack = {str(i): VisionMerger(dim, output_dim, eps, True) for i in (5, 11, 17)}


class Qwen3Vision:
  def __init__(self, kv:dict):
    self.dim = kv['clip.vision.embedding_length']
    self.output_dim = kv['clip.vision.projection_dim']
    self.patch_size = kv['clip.vision.patch_size']
    self.n_heads = kv['clip.vision.attention.head_count']
    self.merge_size = kv['clip.vision.spatial_merge_size']
    self.eps = kv['clip.vision.attention.layer_norm_epsilon']
    self.v = _VisionWeights(self.dim, kv['clip.vision.feed_forward_length'], self.output_dim,
                            kv['clip.vision.block_count'], self.n_heads, self.patch_size, self.eps)
    self.mm = [VisionMerger(self.dim, self.output_dim, self.eps, False)]

  @staticmethod
  def from_gguf(path:str|Path) -> Qwen3Vision:
    kv, state = gguf_load(path)
    model = Qwen3Vision(kv)
    renamed = {}
    for name, value in state.items():
      if name == 'v.patch_embd.weight': name = 'v.patch_embd.weight0'
      elif name == 'v.patch_embd.weight.1': name = 'v.patch_embd.weight1'
      if name.startswith('v.blk.') and ('.attn_qkv.' in name or '.attn_out.' in name):
        parts = name.split('.')
        name = '.'.join(parts[:3] + ['attn'] + parts[3:])
      if name.startswith('v.post_ln.'):
        name = name.replace('v.post_ln.', 'mm.0.norm.')
      elif name.startswith('mm.0.'):
        name = name.replace('mm.0.', 'mm.0.fc1.')
      elif name.startswith('mm.2.'):
        name = name.replace('mm.2.', 'mm.0.fc2.')
      if name.startswith('v.deepstack.'):
        parts = name.split('.')
        name = '.'.join(parts[:3] + ['norm' if parts[3] == 'norm' else parts[3]] + parts[4:])
      renamed[name] = value.cast(dtypes.half)
    nn.state.load_state_dict(model, renamed, verbose=False, consume=True, realize=False)
    return model

  @staticmethod
  def _block_order(x:Tensor, height:int, width:int) -> Tensor:
    return x.reshape(height//2, 2, width//2, 2, x.shape[-1]).permute(0, 2, 1, 3, 4).reshape(height*width, x.shape[-1])

  def _position_embeddings(self, height:int, width:int, device:str|tuple[str, ...]|None) -> tuple[Tensor, Tensor, Tensor]:
    table = self.v.position_embd.weight.reshape(48, 48, self.dim).permute(2, 0, 1).unsqueeze(0)
    pos = table.interpolate((height, width), mode='linear', align_corners=True).squeeze(0).permute(1, 2, 0)
    pos = self._block_order(pos, height, width)
    rows = Tensor.arange(height).to(device).reshape(height, 1).expand(height, width)
    cols = Tensor.arange(width).to(device).reshape(1, width).expand(height, width)
    coords = self._block_order(rows.stack(cols, dim=-1), height, width)
    inv = 1.0 / (10000.0 ** (Tensor.arange(0, self.dim//self.n_heads//2, 2).to(device) / (self.dim//self.n_heads//2)))
    freq = coords.unsqueeze(-1).float() * inv
    freq = freq[:, 0].cat(freq[:, 1], dim=-1)
    return pos, freq.cos().cat(freq.cos(), dim=-1), freq.sin().cat(freq.sin(), dim=-1)

  def __call__(self, image:Tensor) -> tuple[Tensor, list[Tensor], tuple[int, int, int]]:
    if image.ndim != 4 or image.shape[0] != 1 or image.shape[1] != 3: raise ValueError('expected one NCHW RGB image')
    height, width = image.shape[2]//self.patch_size, image.shape[3]//self.patch_size
    if height % 2 or width % 2: raise ValueError('image dimensions must be multiples of 32')
    x = self._block_order(self.v.patch_embd(image), height, width)
    pos, cos, sin = self._position_embeddings(height, width, image.device)
    x = x + pos.cast(x.dtype)
    deepstack = []
    for i, block in enumerate(self.v.blk):
      x = block(x, cos, sin)
      if str(i) in self.v.deepstack: deepstack.append(self.v.deepstack[str(i)](x))
    return self.mm[0](x), deepstack, (1, height, width)


class Qwen3VL(Transformer):
  def __init__(self, config:TransformerConfig, sections:tuple[int, int, int]):
    super().__init__(config)
    self.sections = sections
    self.decode_jit = TinyJit(self._decode)

  def _mrope(self, positions:Tensor) -> Tensor:
    # positions is (3, T); Qwen3-VL interleaves temporal/height/width frequencies in groups of three.
    inv = 1.0 / (self.blk[0].config.rope_theta ** (Tensor.arange(0, self.blk[0].config.rope_dim, 2).to(positions.device) /
                                                   self.blk[0].config.rope_dim))
    freq = positions.unsqueeze(-1).float() * inv
    mixed = freq[0]
    for dim, offset in ((1, 1), (2, 2)):
      length = self.sections[dim] * 3
      indices = Tensor.arange(freq.shape[-1]).to(positions.device)
      mask = (indices % 3 == offset) & (indices < length)
      mixed = mask.where(freq[dim], mixed)
    return mixed.cos().cat(mixed.sin(), dim=-1)

  def _run_hidden(self, x:Tensor, start_pos:int|UOp, freqs:Tensor, image_range:tuple[int, int]|None=None,
                  deepstack:list[Tensor]|None=None) -> Tensor:
    for i, block in enumerate(self.blk):
      x = block(x, start_pos, freqs)
      if deepstack is not None and image_range is not None and i < len(deepstack):
        lo, hi = image_range
        x = x[:, :lo].cat(x[:, lo:hi] + deepstack[i].unsqueeze(0), x[:, hi:], dim=1)
    return x

  def _sample(self, x:Tensor, temperature:Tensor) -> Tensor:
    logits = self.output(self.output_norm(x[:, -1:]))[:, -1, :]
    return (logits / temperature.maximum(1e-12) - (Tensor.rand_like(logits).maximum(1e-12).log().neg()).log()).argmax(-1, keepdim=True)

  def _decode(self, token:Tensor, start_pos:int|UOp, position:int|UOp, temperature:Tensor) -> Tensor:
    freqs = self._mrope(Tensor.full((3, 1), position, dtype=dtypes.int32, device=token.device))
    return self._sample(self._run_hidden(self.token_embd(token).float(), start_pos, freqs), temperature)

  def generate_vision(self, tokens:list[int], image_range:tuple[int, int], image_embeds:Tensor, deepstack:list[Tensor],
                      grid:tuple[int, int, int], max_new_tokens:int=32, temperature:float=0.0):
    lo, hi = image_range
    if hi-lo != image_embeds.shape[0]: raise ValueError('image token count does not match vision embeddings')
    positions, current = [], 0
    for start, end, is_image in ((0, lo, False), (lo, hi, True), (hi, len(tokens), False)):
      if not is_image:
        vals = Tensor.arange(current, current+end-start, dtype=dtypes.int32)
        positions.append(vals.unsqueeze(0).expand(3, end-start))
        current += end-start
      else:
        _, gh, gw = grid
        h = Tensor.arange(gh//2, dtype=dtypes.int32).reshape(gh//2, 1).expand(gh//2, gw//2).flatten()
        w = Tensor.arange(gw//2, dtype=dtypes.int32).reshape(1, gw//2).expand(gh//2, gw//2).flatten()
        positions.append(Tensor.full((hi-lo,), current, dtype=dtypes.int32).stack(h+current, w+current))
        current += max(gh, gw)//2
    pos = positions[0].cat(*positions[1:], dim=1)
    token_tensor = Tensor(tokens, dtype=dtypes.int32).reshape(1, -1)
    x = self.token_embd(token_tensor).float()
    x = x[:, :lo].cat(image_embeds.unsqueeze(0), x[:, hi:], dim=1)
    temp = Tensor([temperature])
    out = self._sample(self._run_hidden(x, 0, self._mrope(pos), image_range, deepstack), temp).realize()
    next_position, start_pos = int(pos.max().item())+1, len(tokens)
    v_start = UOp.variable('vl_start', 0, self.max_context-1)
    v_position = UOp.variable('vl_position', 0, self.max_context-1)
    for i in range(max_new_tokens):
      token = int(out.item())
      yield token
      if i == max_new_tokens-1: break
      out = self.decode_jit(out, v_start.bind(start_pos), v_position.bind(next_position), temp).realize()
      start_pos, next_position = start_pos+1, next_position+1


class Qwen3VLRunner:
  """Fixed prompt and output budget, captured as one GPU graph; no per-token host reads.

  Construct after materializing model weights. The first two calls compile/capture;
  measure steady-state latency from the third call, including input upload/output read.
  The caller truncates the returned tokens at EOS (extra computed tokens are ignored).
  """
  def __init__(self, model:Qwen3VL, vision:Qwen3Vision, tokens:list[int], image_range:tuple[int, int],
               grid:tuple[int, int, int], max_new_tokens:int=16, allowed_tokens:list[int]|None=None):
    if not 1 <= max_new_tokens <= model.max_context-len(tokens): raise ValueError('invalid output budget')
    lo, hi = image_range
    _, gh, gw = grid
    if hi-lo != gh*gw//4: raise ValueError('image token count does not match grid')
    self.model, self.vision = model, vision
    if allowed_tokens is not None and (not allowed_tokens or len(set(allowed_tokens)) != len(allowed_tokens) or
                                      any(t < 0 or t >= model.output.weight.shape[0] for t in allowed_tokens)):
      raise ValueError('allowed_tokens must be distinct vocabulary IDs')
    self.allowed_tokens = Tensor(allowed_tokens, dtype=dtypes.int32).realize() if allowed_tokens is not None else None
    self.output = model.output
    if allowed_tokens is not None:
      # Project only the selected rows: full-vocabulary projection and gather can
      # materialize a float32 vocabulary-sized temporary in the captured graph.
      self.output = Linear(model.output.in_features, len(allowed_tokens), bias=model.output.bias is not None)
      def select_rows(weight:Tensor) -> Tensor:
        rows = [weight[t:t+1] for t in allowed_tokens]
        return rows[0].cat(*rows[1:], dim=0).contiguous().realize()
      self.output.weight = select_rows(model.output.weight)
      if model.output.bias is not None: self.output.bias = select_rows(model.output.bias)
      self.output.resident_fp16 = model.output.resident_fp16
    self.image_range, self.max_new_tokens = image_range, max_new_tokens
    self.prompt_len = len(tokens)
    self.next_position = lo+max(gh, gw)//2+len(tokens)-hi
    positions = [list(range(lo)) for _ in range(3)]
    for dim in range(3):
      positions[dim] += [lo + (0 if dim == 0 else h if dim == 1 else w) for h in range(gh//2) for w in range(gw//2)]
      positions[dim] += list(range(lo+max(gh, gw)//2, self.next_position))
    self.freqs = model._mrope(Tensor(positions, dtype=dtypes.int32)).realize()
    self.embeddings = model.token_embd(Tensor([tokens], dtype=dtypes.int32)).float().realize()
    self.decode_freqs = [model._mrope(Tensor.full((3, 1), self.next_position+i, dtype=dtypes.int32)).realize()
                        for i in range(max_new_tokens-1)]
    self.jit = TinyJit(self.forward)

  def _greedy(self, x:Tensor) -> Tensor:
    logits = self.output(self.model.output_norm(x[:, -1:]))[:, -1]
    if self.allowed_tokens is not None:
      return self.allowed_tokens[logits.argmax(-1, keepdim=True)]
    return logits.argmax(-1, keepdim=True)

  def forward(self, image:Tensor) -> Tensor:
    embeds, deepstack, _ = self.vision(image)
    # Realization boundaries bound compilation memory; TinyJit combines the launches.
    Tensor.realize(embeds, *deepstack)
    lo, hi = self.image_range
    x = self.embeddings[:, :lo].cat(embeds.unsqueeze(0), self.embeddings[:, hi:], dim=1)
    x = self.model._run_hidden(x, 0, self.freqs, self.image_range, deepstack)
    out = self._greedy(x).realize()
    outputs = [out]
    for i, freqs in enumerate(self.decode_freqs):
      x = self.model.token_embd(out).float()
      x = self.model._run_hidden(x, self.prompt_len+i, freqs)
      out = self._greedy(x).realize()
      outputs.append(out)
    return outputs[0].cat(*outputs[1:], dim=1).realize()

  def __call__(self, image:Tensor) -> Tensor:
    return self.jit(image)


def materialize_weights(model, fp16_compute:bool=False) -> None:
  """Decode GGUF weights once into VRAM instead of on every matmul (fits the 2B model on 8 GB)."""
  seen = {}
  for weight in nn.state.get_parameters(model):
    key = weight.uop
    if key not in seen: seen[key] = weight.contiguous().realize()
    weight.replace(seen[key])
  if fp16_compute:
    for layer in nn.state.get_state_dict(model, tensor_type=Linear).values():
      assert isinstance(layer, Linear)
      if layer.weight.dtype != dtypes.half: raise ValueError('FP16 compute requires half weights')
      layer.resident_fp16 = True
