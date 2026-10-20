# CPU vs GPU: Reversible Image Pipeline

Heterogeneous Computing Lab, GPU Tutorial (Part 2/3)

A set of video frames was obfuscated by a deterministic, reversible PyTorch pipeline (rolls, XOR, modular addition, channel permutation). The task is to implement the inverse pipeline, reconstruct the original frames, compare CPU and GPU runtime, and turn the frames back into a video.

## Setup

### 1. Clone (or pull) repository

```
git clone https://github.com/spetrescu/cpu-gpu-example.git
```

or, if you have the repo already, pull with `git pull`.

### 2. Create a virtual environment

```
python3 -m venv venv
```

Activate it:

```
source venv/bin/activate
```

### 3. Install dependencies

```
pip install numpy opencv-python torch Pillow
```

## Your Task

You must complete the TODOs in: `inverse_batch_exercise.py`. Subsequently, you must run the file (see below guidance on how to run it and with which flags) and collect measurements for CPU vs CUDA enabled acceleration.

## My Solution

Each forward substep applies, in this order: vertical roll → horizontal roll → XOR with a mask → addition modulo 256 → channel permutation. To invert it, every operation is undone **in reverse order** (the stages and substeps are already iterated in reverse by the provided code):

```python
# TODO 1: undo channel permutation
x = x[..., inv_perm]

# TODO 2: undo modular addition
x = ((x.to(torch.int16) - add_s) & 255).to(torch.uint8)

# TODO 3: recreate same mask and undo XOR
mask = make_mask(yy, xx, cc, xor_seed_s)
x = torch.bitwise_xor(x, mask)

# TODO 4: undo horizontal and vertical rolls
x = torch.roll(x, shifts=-dx_s, dims=2)
x = torch.roll(x, shifts=-dy_s, dims=1)
```

- **Channel permutation:** indexing with the precomputed inverse permutation puts every channel back in its original position.
- **Modular addition:** subtract the same value. The cast to `int16` avoids underflow, and `& 255` wraps the result back into 0–255 (same as mod 256).
- **XOR:** XOR is its own inverse (`(a ^ m) ^ m = a`), so applying the same mask again removes it. `make_mask` is deterministic, so the same seed gives the same mask.
- **Rolls:** rolling by the negative shift moves every pixel back. Horizontal is undone first because it was applied last.

## Running

```
python inverse_batch_exercise.py --input_dir frames --device cuda --save_outputs
python inverse_batch_exercise.py --input_dir frames --device cpu --save_outputs
```

## Results: CPU vs GPU

![Compute time per batch, CPU vs GPU](cpu_vs_gpu.png)

For the first four batches (32 images each), the GPU needs about 6 s per batch while the CPU needs about 210–216 s. That is roughly a **35× speedup** for the same code, only the device changed. The work is embarrassingly parallel: every pixel and every image is transformed independently, so the GPU can process them all at once, while the CPU works through far fewer elements in parallel.

## Reconstructed Video

The reconstructed frames were merged into a 24 fps video with ffmpeg:

```
ffmpeg -framerate 24 -i out_reconstructed/frame_%05d_reconstructed.png -c:v libx264 -pix_fmt yuv420p reconstructed.mp4
```

<video src="reconstructed.mp4" controls width="640"></video>

If the video does not play, [download it here](reconstructed.mp4).
