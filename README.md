# Gravity — CUDA Barnes-Hut 3D N-body

A real-time 3D gravitational N-body simulation on the GPU. Forces are computed
with a **Barnes-Hut** approximation built on a **lock-free LBVH** (Morton sort +
Karras radix tree + atomic bottom-up centre-of-mass), and the particles are
rendered live through **CUDA↔OpenGL interop** as additively-blended point sprites.

Handles 100k particles in real time and scales to millions.

## Measured on an RTX 5060 (sm_120), θ=0.5

| Bodies | ms/step | FPS (compute) |
|-------:|--------:|--------------:|
| 100,000 | 17.1 | ~58 |
| 1,000,000 | 247 | ~4 |

Accuracy vs brute-force O(N²): mean relative acceleration error **~1.2%**.

## Build (Windows)

Requires: CUDA Toolkit 12.x, CMake ≥ 3.24, Git, MSVC Build Tools 2022.
GLFW and GLAD are fetched automatically by CMake.

```bat
build.bat
```

(or manually, from an x64 Native Tools prompt:)

```bat
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

For a different GPU, override the arch, e.g. `-DCMAKE_CUDA_ARCHITECTURES=89`.

## Run

```bat
build\gravity.exe                  REM interactive 3D viewer (100k bodies)
build\gravity.exe --n=300000       REM choose body count
build\gravity.exe --bench          REM headless: print ms/step and FPS
build\gravity.exe --verify         REM compare Barnes-Hut vs brute force
```

Extra flags: `--steps=N`, `--theta=0.6`, `--dt=0.0008`, `--gas=N`, `--dust=N`,
`--nogas` (gas/dust are viewer-only; `--bench`/`--verify` stay pure-star),
`--headless`, `--fps=N`, `--substeps=N` (see [Headless rendering](#headless-rendering-faster-than-real-time)).

> The viewer opens a GLFW window — run it from a normal terminal (or
> double-click the exe). Launching it from an automated/background context
> may not surface the window.

### Viewer controls
Free-fly camera — full 360° look, no gimbal lock.
- **Left-drag** — look around
- **WASD** — move (forward/left/back/right), **Q/E** — down/up
- **Shift** — move faster · **Scroll** — dolly forward/back
- **Space** — pause
- **X** — toggle auto-spin
- **V** — toggle velocity colouring (blue = slow → red = fast)
- **N** — toggle nebulae + dust (colourful emission gas and dark dust lanes)
- **C** — start / stop video recording → `gravity_capture_<timestamp>.mp4`
- **1–7** — load an initial-condition preset (rebuilds the simulation live)
- **R** — reset camera
- **Esc** — quit

### Presets (keys 1–7, or `--preset=N`)
1. **Spiral galaxy** — single rotating disk with a bright bulge
2. **Galaxy collision** — two equal disks on a grazing encounter
3. **Minor merger** — large galaxy + infalling satellite
4. **Cold collapse** — uniform sphere falls in and fragments
5. **Expanding cloud** — sphere with outward (Hubble-like) velocity
6. **Head-on collision** — two perpendicular disks meeting head-on
7. **Big Bang** — cosmological structure formation (see below)

#### Big Bang / galaxy formation (preset 7)
A near-uniform sphere is given **Zel'dovich density perturbations** (a sum of
random Fourier displacement modes with a red, ~1/k spectrum) and a **Hubble
expansion** velocity field `v = H₀·x` plus the matching growing-mode peculiar
velocity. `H₀` is set just **below the binding value** `√(2GM/R³)`, so the volume
expands, turns around, and gravity amplifies the seed perturbations: matter drains
out of voids into **sheets → filaments → collapsed haloes** (proto-galaxies) that
then **merge hierarchically** — the same qualitative process as ΛCDM structure
formation, in pure Newtonian gravity. This preset runs at **1,000,000 particles**
with **no nebulae/dust** (pure stars) so the cosmic web reads clearly; override
with `--n=N`. Knobs live at the top of preset 7 in `barnes_hut.cu`: `H0` (lower =
collapses sooner, higher = flies apart), the perturbation amplitude `amp`, and
mode count `NM`.

Stellar masses follow a **Salpeter IMF** (`dN/dm ∝ m^-2.35`, 0.1–50 M☉): most
stars are light, a few are very massive — producing realistic clumpy gravity.

### Rendering
Stars are drawn as additive sprites into an **HDR (RGBA16F)** buffer, then
tone-mapped with **ACES**. Each star has an intrinsic **black-body colour** (cool
red → white → hot blue) and luminosity, with a bright central bulge and rare giants.
A subtle **bloom** glow is applied in realistic mode only; **velocity mode renders
crisp points with no glow**.

**Nebulae and dust** (toggle **N**, realistic mode only) are **simulated**, not a
backdrop. They are extra **low-mass tracer bodies** seeded into the same disks /
collapse spheres as the stars, with matching orbital velocities. They feel the
full Barnes-Hut gravitational field but carry negligible mass (~2×10⁻⁷ each), so
they are pushed around by the stars without perturbing them — and therefore
**shear into spiral lanes, tidal tails and clumps on their own** as the system
evolves. Body layout in the shared buffer is `[stars | gas | dust]`, drawn as
three index ranges.

Gas is drawn as small soft **additive** sprites; emission colour comes from a
**coherent 3D value-noise field** evaluated in world space (H-α red, magenta,
reflection blue, cyan, O-III teal, dusty gold) so neighbouring parcels share a hue
— real nebula regions rather than confetti. **Dust** is a second range of dark,
slightly warm sprites blended **alpha-over**, carving mottled **dark lanes** into
the glow before the stars are drawn on top. At birth the gas/dust is biased onto a
**2-arm logarithmic spiral** (dust tightest, razor-thin) so it hugs the arms like
real galactic gas rather than filling the disk. Counts default to a modest ⅕N gas
+ ⅒N dust; tune with `--gas=N` / `--dust=N`, or `--nogas`.

### Recording
Press **C** to record. The pipeline renders into an offscreen **1920×1080** target,
so the video is always at least 1080p regardless of the window size. Frames are
piped to **ffmpeg** and muxed at a fixed **60 fps**, so each
`gravity_capture_<timestamp>.mp4` is always smooth 60 fps no matter how fast the
simulation renders. The `--record` / `--recframes=N` flags do the same from the
command line. Requires ffmpeg on `PATH`:

```bat
winget install Gyan.FFmpeg
```

#### Headless rendering (faster than real time)
`--headless` opens a **hidden** OpenGL context with **vsync off**, so the
simulation is no longer capped at the 60 fps display rate — it renders frames as
fast as the GPU can compute them and pipes them straight to ffmpeg, then exits.
It implies recording; `--recframes=N` sets how many frames to write (default
1800 = 30 s of video). No window appears.

```bat
REM 1M-body Big Bang, 1200 frames, 4 sim steps per frame, 60 fps video
build\gravity.exe --headless --preset=7 --recframes=1200 --substeps=4
```

Extra recording flags:
- `--fps=N` — output video frame rate (default 60). Each rendered frame is one
  video frame, so a higher value plays the same motion back faster.
- `--substeps=N` — advance the simulation **N** steps per rendered frame (default
  1). Lets the system evolve faster per second of video without shrinking `--dt`
  (which would cost accuracy); great for watching the slow Big Bang collapse.

## How it works (per step)

1. `bounds` — enclosing cube of all bodies (reduction).
2. `morton` — 30-bit Z-order key per body.
3. `sort` — order bodies along the space-filling curve (thrust).
4. `radix tree` — Karras 2012 binary tree over the sorted keys (parallel, lock-free).
5. `summarize` — bottom-up COM + AABB via atomic arrival counters (no spin-locks).
6. `forces` — per-body tree traversal; a subtree is approximated by its COM when
   `size² < θ²·dist²`. Bodies are traversed in Morton order to minimise warp divergence.
7. `integrate` — symplectic (kick-drift) Euler.

## Files
- `src/sim.cuh` — host-side `Simulation` interface.
- `src/barnes_hut.cu` — all CUDA kernels + the simulation driver.
- `src/main.cpp` — OpenGL viewer, camera, and the `--bench` / `--verify` modes.
- `CMakeLists.txt`, `build.bat` — build.
