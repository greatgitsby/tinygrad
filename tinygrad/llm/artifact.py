"""Local, trusted compiled-JIT artifacts with streamed, deduplicated device buffers.

Pickles execute code: never load artifacts from an untrusted source. The caller
must invalidate the cache when the program, weights, device or runtime changes.
"""
from __future__ import annotations
import json, os, pathlib, pickle, shutil, tempfile, time
from tinygrad import Device, dtypes
from tinygrad.device import Buffer

CHUNK_SIZE = 32 << 20


def save_artifact(obj, path:str|pathlib.Path) -> None:
  """Publish an immutable directory atomically, without holding all weights in RAM."""
  path = pathlib.Path(path)
  if path.exists(): raise FileExistsError(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = pathlib.Path(tempfile.mkdtemp(prefix=f'.{path.name}.', dir=path.parent))
  try:
    offsets:dict[Buffer, int] = {}
    with (temporary/'buffers.bin').open('wb') as data, (temporary/'program.pkl').open('wb') as program:
      def persistent_id(buf):
        if not isinstance(buf, Buffer) or buf.device != Device.DEFAULT or not buf.is_allocated(): return None
        base = buf.base
        if base not in offsets:
          data.write(bytes((-data.tell()) % 256))
          offsets[base] = data.tell()
          if shutil.disk_usage(temporary).free < base.nbytes + (256 << 20): raise OSError('insufficient space for artifact')
          for off in range(0, base.nbytes, CHUNK_SIZE):
            size = min(CHUNK_SIZE, base.nbytes-off)
            data.write(base.view(size, dtypes.uint8, off).ensure_allocated().as_memoryview())
        return buf.size, buf.dtype, offsets[base]+buf.offset
      class Writer(pickle.Pickler):
        def persistent_id(self, obj): return persistent_id(obj)
      Writer(program, protocol=5).dump(obj)
      for f in (data, program):
        f.flush()
        os.fsync(f.fileno())
    manifest = {'version': 1, 'device': Device.DEFAULT, 'buffers_bytes': (temporary/'buffers.bin').stat().st_size,
                'program_bytes': (temporary/'program.pkl').stat().st_size}
    with (temporary/'manifest.json').open('w') as manifest_file:
      json.dump(manifest, manifest_file)
      manifest_file.flush()
      os.fsync(manifest_file.fileno())
    temporary.rename(path)
  except BaseException:
    shutil.rmtree(temporary)  # only this call's unpublished, newly-created temporary directory
    raise


def load_artifact(path:str|pathlib.Path) -> tuple[object, dict[str, float]]:
  """Upload a memory-mapped weight arena, then restore the already-compiled graph."""
  path = pathlib.Path(path)
  start = time.perf_counter()
  manifest = json.loads((path/'manifest.json').read_text())
  if manifest['version'] != 1 or manifest['device'] != Device.DEFAULT: raise ValueError('incompatible artifact')
  for name, size in (('buffers.bin', manifest['buffers_bytes']), ('program.pkl', manifest['program_bytes'])):
    if (path/name).stat().st_size != size: raise ValueError(f'truncated artifact: {name}')
  arena = Buffer(Device.DEFAULT, manifest['buffers_bytes'], dtypes.uint8, preallocate=True)
  source = Buffer(f'DISK:{path/"buffers.bin"}', arena.size, dtypes.uint8, preallocate=True)
  for off in range(0, arena.nbytes, CHUNK_SIZE):
    size = min(CHUNK_SIZE, arena.nbytes-off)
    arena.view(size, dtypes.uint8, off).ensure_allocated().copy_from(source.view(size, dtypes.uint8, off).ensure_allocated())
  Device[Device.DEFAULT].synchronize()
  uploaded = time.perf_counter()
  views = {}
  def persistent_load(pid):
    size, dtype, offset = pid
    if offset < 0 or size < 0 or offset+size*dtype.itemsize > arena.nbytes: raise ValueError('invalid artifact buffer')
    if pid not in views: views[pid] = arena.view(size, dtype, offset)
    return views[pid]
  with (path/'program.pkl').open('rb') as f:
    class Reader(pickle.Unpickler):
      def persistent_load(self, pid): return persistent_load(pid)
    obj = Reader(f).load()
  return obj, {'upload_s': uploaded-start, 'deserialize_s': time.perf_counter()-uploaded, 'bytes': arena.nbytes}
