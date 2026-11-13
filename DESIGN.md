# SprayBot — Technical Design Specification

## 1. Overview & Goals

**SprayBot** is a native Linux 3D robotics simulator and offline video renderer written in **Rust + Bevy**. It simulates a full-scale 6-DOF industrial robot arm mounted on a 1-DOF linear floor track (7-DOF total) inside an industrial mural studio. Given an arbitrary input **SVG**, the robot picks up matching aerosol spray cans from a rack at the end of the track, shakes them, and paints the vector artwork onto a large wall-mounted canvas—tracing lines with variable standoff width and filling regions via contour-then-hatch toolpaths.

### Core Design Pillars
- **Pure-code Bevy architecture** with decoupled kinematics and visuals so procedural PBR robot links can be swapped for URDF/glTF industrial arm meshes later without touching solver or planner code.
- **Zero-GUI immersive viewport** controlled entirely via `clap` CLI flags, mouse camera controls, and keyboard shortcuts.
- **Physically grounded spray deposition** combining cone-wall geometry, speed-compensated paint accumulation, edge speckle, volumetric mist particles, and dynamic PBR canvas roughness/thickness maps.
- **Deterministic fixed-timestep video export** piping exact offscreen/readback frames directly to `ffmpeg` (MP4) or PNG sequences without dropped frames.

### Verified Host Environment
- **OS:** Linux (Fedora 44)
- **Rust Toolchain:** `rustc 1.99+` / Cargo
- **Video Encoder:** `ffmpeg 8.1+` available on `$PATH`

---

## 2. World Coordinate System & Studio Layout

Bevy's standard right-handed Y-up coordinate system is used throughout (`1 unit = 1.0 meter`):

- **Origin `(0, 0, 0)`:** Floor level directly beneath the horizontal center of the canvas wall.
- **Axes:**
  - `+X`: Right along the wall / linear track
  - `+Y`: Up toward warehouse ceiling
  - `+Z`: Outward from the wall toward the viewer / robot track

### Physical Dimensions

| Element | Dimensions / Pose | Notes |
| :--- | :--- | :--- |
| **Studio Room** | `8.0m (X) × 4.2m (Y) × 6.5m (Z)` | Concrete floor (scuffed PBR roughness), industrial brick/concrete back wall (`Z = -0.04m`), overhead skylight + angled studio spotlights with soft shadows |
| **Wall Canvas** | `2.50m (W) × 1.50m (H) × 0.04m (D)` | Front painting plane lies at `Z = 0.00m`, centered at `(0.0, 1.55, 0.0)` (`X ∈ [-1.25, +1.25]`, `Y ∈ [0.80, 2.30]`) |
| **Linear Floor Track** | `3.80m` rail centered along `Z = 0.95m` | Carriage travel `q_0 ∈ [-1.75m, +1.45m]` at height `Y = 0.18m` |
| **6-DOF Robot Arm** | `~1.35m` max reach | Mounted atop the rail carriage; comfortably reaches entire `2.5m × 1.5m` canvas plus the side spray rack |
| **Spray Can Rack** | Positioned near `X = -1.95m, Z = 0.55m` | 2-tier angled steel shelf holding up to `--max-cans` (default `16`) color-coded aerosol cans (`Ø 0.066m × 0.19m`) within easy reach when `q_0 = -1.75m` |

---

## 3. High-Level Architecture & Workspace Boundaries

To maximize testability and enable parallel multi-agent implementation, pure geometry/planning/math logic is cleanly separated from Bevy ECS/rendering systems:

```text
spraybot/
├── Cargo.toml
├── DESIGN.md
├── PLAN.md
├── assets/
│   └── examples/          # Built-in test SVGs (geometric, multi-layer mural, typography)
└── src/
    ├── main.rs            # CLI parsing & Bevy App entry point
    ├── cli.rs             # clap arguments, resolution/export config
    ├── core/              # Pure Rust (NO Bevy ECS dependencies — fast unit tests!)
    │   ├── mod.rs
    │   ├── svg_ingest.rs  # usvg + fontdb parsing, flattening, winding -> geo shapes
    │   ├── palette.rs     # OKLab color extraction & k-means rack quantization
    │   ├── batching.rs    # Overlap-aware topological DAG color scheduler
    │   ├── planner.rs     # Contour + boustrophedon hatch toolpaths, dynamic Z-standoff, lead-in/out
    │   ├── kinematics.rs  # Data-driven 7-DOF chain, FK, Weighted DLS Jacobian IK
    │   └── trajectory.rs  # S-curve / trapezoidal velocity profiler & state-machine timeline
    └── sim/               # Bevy ECS plugins, PBR scene, GPU/CPU canvas, camera, video export
        ├── mod.rs
        ├── environment.rs # Warehouse room, PBR concrete/steel materials, studio lighting
        ├── robot_view.rs  # Procedural PBR link meshes bound to KinematicChain + can rack visuals
        ├── controller.rs  # Playback clock, state machine execution, can attachment & shake
        ├── canvas.rs      # Dynamic RGBA albedo + roughness/thickness PBR textures & spray splatting
        ├── mist.rs        # Volumetric spray cone billboard/particle emitter
        ├── camera.rs      # Orbit/pan/zoom mouse camera + hotkeys & follow-nozzle presets
        └── recorder.rs    # Fixed-timestep GPU frame readback -> ffmpeg MP4 pipe or PNG dir
```

---

## 4. SVG Ingestion, Palette & Toolpath Pipeline (`src/core/`)

### 4.1 SVG Parsing & Feature Normalization (`svg_ingest.rs`)
- **Crate stack:** `usvg` (with `fontdb` loading system fonts so `<text>` elements automatically convert to vector outlines), `kurbo` / `lyon_geom` for adaptive Bézier flattening (`tolerance = 0.5mm` in canvas space), and `geo` / `i_overlay` for robust 2D boolean/winding polygon operations.
- **Canvas Mapping:** Fit the SVG `viewBox` uniformly into the canvas (`2.50m × 1.50m`) minus a configurable margin (default `0.08m` per side), preserving aspect ratio and centering on `(0.0, 1.55)`. Flip SVG `+Y` (down) to world `+Y` (up).
- **Supported Primitives:**
  - Paths, rects, circles, ellipses, polygons, polylines, groups/transforms, compound paths with holes (`evenodd` and `nonzero` fill rules normalized into `geo::MultiPolygon<f64>`).
  - `<text>` converted to outlines via `usvg`'s `fontdb` database.
  - **Opacity:** Element/fill/stroke alpha `α ∈ (0, 1]` maps to target paint coverage density (modulating nozzle feed speed / flow rate multiplier).
  - **Gradients (v1 simplified):** `<linearGradient>` and `<radialGradient>` fills are either quantized into `K` (`4–6`) clipped iso-band polygons along the gradient axis/radius using stop colors, or collapsed to their area-weighted stop color if sub-threshold in size.

### 4.2 Dynamic Palette & Rack Quantization (`palette.rs`)
- Collect all distinct solid colors referenced across normalized fills and strokes.
- Convert sRGB colors into perceptual **OKLab** (`palette` or lightweight inline math).
- Merge near-identical colors within perceptual threshold `ΔE_oklab < 0.015`.
- If unique colors `<= max_cans` (default `16`), spawn exact matching spray cans on the rack (leaving unused rack slots empty).
- If unique colors `> max_cans`, run deterministic **k-means++** (seeded for reproducibility, weighted by total painted area/arc-length so dominant colors get exact cans) to quantize down to `max_cans` palette entries, remapping every primitive to its nearest centroid.

### 4.3 Overlap-Aware Topological Color Batching (`batching.rs`)
Painting strictly in SVG order causes excessive rack swaps, while naive grouping by color causes background shapes to paint *over* foreground details.
1. Assign each drawable primitive an index `0..N` in back-to-front SVG painter's order.
2. Build a directed dependency graph (DAG): add edge `i -> j` (`i < j`) if primitive `i` and primitive `j` have **different palette colors** AND their expanded painted footprints (`AABB` pre-check + `geo` polygon/buffered-stroke intersection expanded by spray feather margin `~8mm`) overlap.
3. Schedule batches greedily:
   - Maintain the frontier of primitives with in-degree `0`.
   - While primitives matching `current_color` exist in the frontier, emit all of them (ordered by nearest-neighbor travel heuristic `2-opt`/greedy to minimize air moves!).
   - When exhausted, pick the next color in the frontier that unlocks or batches the largest group of ready primitives, insert a `SwapCan(color_id)` command, and repeat.

### 4.4 Spray Cone Geometry & Toolpath Generation (`planner.rs`)
- **Physical Nozzle Model:**
  - Spray cone half-angle: `θ_c = 12°` (`tan θ_c ≈ 0.2125`).
  - Standoff distance from wall (`Z`): clamped to `d ∈ [0.04m, 0.26m]`.
  - Effective core spot width: `w(d) = 2 * d * tan(θ_c)`, spanning **`~1.7cm` (at `4cm` standoff)** up to **`~11.0cm` (at `26cm` standoff)**.
- **Stroke Toolpaths:**
  - If scaled SVG `stroke_width <= w(d_max)`: trace a single centerline pass at standoff `d = clamp(stroke_width / (2 * tan(θ_c)), d_min, d_max)`.
  - If `stroke_width > w(d_max)`: buffer the stroke polyline into a polygon (`geo` offset/buffer) and treat it as a filled region.
- **Filled Polygon Toolpaths (Contour-then-Hatch):**
  1. **Perimeter Pass:** Inset the polygon (outer ring inward, hole rings outward) by `0.45 * w(d_perimeter)` where `d_perimeter = 0.05m` (`w ≈ 2.1cm`). Trace the inset boundary at close standoff for crisp, sharp edges without overspraying outside the shape. (For thin slivers narrower than `w(d_min)`, fall back to medial/skeleton centerline pass at `d_min`).
  2. **Interior Boustrophedon (Zig-Zag) Hatch:** Inset further by `0.6 * w(d_perimeter)` and intersect parallel scanlines spaced at `step_over = 0.65 * w(d_fill)` (35% Gaussian overlap produces uniform flat fill without banding!). Choose `d_fill` dynamically (`0.08m–0.22m`) proportional to the polygon's minor dimension. Connect adjacent scanline segments smoothly when staying inside the polygon; retract and rapid-move across holes/concavities.
- **Speed Compensation & Lead-In / Lead-Out (Anti-Blobbing):**
  - Linear nozzle speed `v_spray` scales inversely with `d * target_opacity` (clamped to `[0.12 m/s, 0.85 m/s]`) so wider standoffs lay down the same paint thickness per pass as narrow ones.
  - Every continuous spray segment prepends a **tangential lead-in** (`L_lead ≈ 0.035m`, valve OFF while accelerating up to `v_spray`), toggles **valve ON** strictly across the path geometry at steady cruising speed, and appends a **tangential lead-out** (`L_lead ≈ 0.035m`, valve OFF before decelerating).

---

## 5. 7-DOF Kinematics, IK & Motion Control (`kinematics.rs`, `trajectory.rs`)

### 5.1 Data-Driven Kinematic Chain (Model-Swap Ready)
Define the robot via a serial `KinematicChain` struct loaded from a clean configuration struct (or TOML/URDF-compatible descriptor):
- **Joint 0 (`RailX`):** `Prismatic` along `+X`, range `[-1.75m, +1.45m]`, max vel `1.2 m/s`, max accel `2.5 m/s²`.
- **Joints 1–6 (`Arm`):** 6 `Revolute` joints (industrial UR10e / KUKA KR-style proportions: base yaw `J1`, shoulder pitch `J2`, elbow pitch `J3`, wrist1 `J4`, wrist2 `J5`, flange roll `J6`; total reach `~1.35m`), max joint vel `2.2–3.8 rad/s`, max accel `6.0–12.0 rad/s²`.
- **Tool Center Point (TCP):** Gripper jaws + held aerosol can offset (`nozzle_origin` and `nozzle_dir` in flange frame).
- **Visual Decoupling:** Each link entity in Bevy holds only a `RobotLink { index: usize }` component whose `Transform` is updated from Forward Kinematics (`FK`). Initially populated with sleek procedural PBR primitives (cast iron dark grey carriage, safety-orange/graphite arm tubes, machined aluminum wrist housings, parallel pneumatic gripper + servo trigger finger); swapping to `.gltf` sub-meshes later only changes `robot_view.rs` spawning.

### 5.2 Weighted Damped Least-Squares (DLS) Jacobian IK with Symmetry Relaxation
When painting on the wall, the spray plume is axially symmetric around the nozzle spray vector `n_tcp` (though keeping the aerosol can roughly upright prevents syphon tube starvation in real cans!).
- **Task Error `e ∈ ℝ⁶`:**
  - Position error `e_pos = p_target - p_tcp` (weight `1.0`).
  - Pointing alignment error (`n_tcp × n_target`, weight `0.8`) + soft upright roll bias keeping the can body within `±35°` of vertical (`+Y`, soft weight `0.15`) so the wrist never hits hard `±π` flip singularities!
  - During rack pick/place, switch roll weight to `1.0` for exact 6-DOF alignment with the rack slot.
- **Weighted DLS Update Step:**
  $$\Delta q = W^{-1} J^T \left(J W^{-1} J^T + \lambda^2 I\right)^{-1} e + \left(I - J^{\#}_W J\right) \alpha (q_{\text{comfort}} - q)$$
  - **Diagonal Weight Matrix `W`:** Heavy weight on `J0` (rail, e.g., `w_0 = 6.0`), moderate on shoulder/elbow (`w_1..3 = 1.5`), light on agile wrist (`w_4..6 = 0.4`). This produces natural emergent coordination: the heavy rail glides smoothly toward the region of interest while the wrist deftly tracks tight curves!
  - **Null-Space Posture `q_comfort`:** Biases the arm into an ergonomic "elbow-up" pose and softly repels joint limits.
  - Run 4–10 Newton/DLS sub-iterations per trajectory sample (seeded from previous configuration `q_{t-1}` for `C¹` continuity).

### 5.3 Choreography State Machine & Velocity Profiling
High-level robot job compiles into a time-parameterized trajectory (`S-curve` / trapezoidal velocity ramps respecting joint & Cartesian velocity/accel limits):
1. `MoveToRackApproach(slot)` → Rapid smooth joint-space + rail glide to safe approach pose in front of `slot`.
2. `PickCan(slot)` → Linear insert (`0.12m`), animate gripper fingers closing (`0.25s`), reparent can entity to gripper flange, lift vertically (`0.06m`) out of slot, retract.
3. `ShakeCan` → Energetic `1.1s` sinusoidal wrist/forearm oscillation (`~6 Hz`, `±22°` pitch/roll) to mix paint before first stroke!
4. `TransitToSegment` → Safe Z-retracted (`Z = 0.30m`) spline move to segment lead-in start.
5. `ExecuteSpraySegment` → Lead-in (valve OFF) → depress servo trigger finger (`valve ON`) along constant-speed path → release trigger (`valve OFF`) on lead-out.
6. `ReturnCan(slot)` → Reverse of pick sequence before picking next color or parking at `HomePose` upon completion.

---

## 6. Realistic Rendering, Canvas PBR & Spray FX (`src/sim/`)

### 6.1 Dual-Texture PBR Canvas (`canvas.rs`)
The wall canvas mesh uses Bevy's `StandardMaterial` backed by dynamically updated textures (default `2048 × 1228` pixels, ~`1.2mm/texel`, configurable via `--canvas-res`):
1. **Base Procedural Canvas Weave:** Normal map (`canvas_normal_tex`) generated at startup with subtle linen cross-weave micro-relief, paired with warm unprimed cotton/gesso off-white base color (`#F4F1EA`).
2. **Dynamic Albedo Texture (`albedo_tex`, `Rgba8UnormSrgb`):** Updated every simulation sub-step when `valve_on == true`.
3. **Dynamic Paint Thickness / Roughness Map (`metallic_roughness_tex`):** Bare canvas starts matte (`perceptual_roughness = 0.88`). As paint thickness `h(u,v)` accumulates, roughness interpolates smoothly down to satin acrylic (`0.32`), subtly catching warehouse specular highlights on painted strokes!

#### Deposition Kernel (Sub-stepped along Nozzle Ray)
To avoid discrete "dotting" at high timelapse speeds (`--speed 10`), each frame sub-steps the nozzle trajectory at spacing `Δs <= 0.25 * spot_radius`:
- Intersect nozzle ray `(p_nozzle, dir_nozzle)` with plane `Z = 0`.
- Compute local spot radius `R = d * tan(θ_c)` and elliptical stretch if `dir_nozzle` is non-perpendicular (`cos φ` incidence factor).
- For texels within `1.35 * R`:
  - Core profile: `falloff = exp(-2.2 * (r / R)^2)`.
  - Outer rim (`0.75 < r/R < 1.35`): modulate with deterministic spatial hash noise to reproduce fine aerosol overspray speckle!
  - Alpha accumulation (Beer-Lambert / Porter-Duff over): `α_step = 1.0 - exp(-flow_rate * falloff * dt_sub / (π * R^2))`, blending can color into `albedo_tex` and incrementing paint thickness in `metallic_roughness_tex`.

### 6.2 Volumetric Spray Mist (`mist.rs`)
- Built directly with lightweight Bevy instanced/billboard soft-alpha quads (avoids fragile third-party GPU particle crate version mismatches and works deterministically in offline video renders!).
- While `valve_on == true`, spawn ~120–200 small soft-lit mist puffs/sec tinted to the active can color inside the `12°` nozzle cone, expanding radially, decelerating via quadratic air drag, and fading out over `0.18–0.35s`.

### 6.3 Studio Lighting & Atmosphere (`environment.rs`)
- Procedural environment/skybox reflection probe (or synthesized warehouse equirectangular HDRI texture generated in memory at startup so metallic robot links and satin paint have rich specular reflections with zero external binary asset bloat!).
- Key directional skylight casting soft PCF shadows + 2 warm/cool overhead warehouse spot lights aimed at the canvas and spray rack + subtle bloom & ACES tonemapping.

---

## 7. CLI, Camera Controls & Video Export (`cli.rs`, `camera.rs`, `recorder.rs`)

### 7.1 Command-Line Interface (`clap`)
```text
spraybot [OPTIONS] [SVG_PATH]

Arguments:
  [SVG_PATH]                 Input SVG file (defaults to built-in showcase mural if omitted)

Options:
  -s, --speed <MULTIPLIER>   Simulation playback speed multiplier [default: 1.0]
  -r, --record <OUTPUT>      Export video (.mp4/.mkv via ffmpeg) or image sequence (directory/)
      --fps <FPS>            Target frame rate for video export [default: 60]
      --resolution <WxH>     Viewport / video render resolution [default: 1920x1080]
      --canvas-res <WIDTH>   Horizontal pixel resolution of canvas texture [default: 2048]
      --max-cans <COUNT>     Spray rack slot capacity before k-means quantization [default: 16]
      --camera <PRESET>      Initial camera preset: overview | follow | rack | canvas [default: overview]
      --headless             Render video offscreen without opening an OS window (requires --record)
      --seed <SEED>          RNG seed for speckle & k-means determinism [default: 42]
```

### 7.2 Mouse & Hotkey Controls (No On-Screen GUI)
- **Mouse:**
  - `Left-drag`: Orbit azimuth / elevation (clamped above floor `Y >= 0.08m` and inside studio walls)
  - `Right-drag` or `Middle-drag`: Pan target focal point
  - `Scroll wheel`: Smooth exponential dolly zoom
- **Keyboard Shortcuts:**
  - `1`: **Studio Overview** (wide 3/4 angle showing canvas, rail, arm, and rack)
  - `2`: **Action / Follow Cam** (smoothly tracks TCP / canvas painting zone with damped spring lag)
  - `3`: **Rack Closeup** (frames the spray can rack and gripper swap choreography)
  - `4`: **Canvas Front** (orthographic-like dead-center framing of the mural)
  - `Space`: Pause / Resume simulation
  - `[` / `]`: Decrease / Increase playback speed (`0.25x` .. `32x`)
  - `G`: Toggle debug 3D gizmos (planned toolpaths, IK target frame, spray cone wireframe)
  - `R`: Reset canvas & restart painting from `t = 0`
  - `Esc` / `Q`: Gracefully finalize any active video recording and exit

### 7.3 Deterministic Fixed-Timestep Video Export (`recorder.rs`)
When `--record <OUTPUT>` is provided:
- **Simulation Clock Lockstep:** Instead of advancing by variable wall-clock `Time<Real>`, each rendered frame advances simulation time by exact `Δt_sim = speed_multiplier / fps`. Even if GPU readback or `4K` rendering runs slower than real-time, every single frame of motion, spray deposition, and mist is `100%` smooth and deterministic.
- **GPU Readback Pipeline:** Render camera to both the window (unless `--headless`) and a GPU readback staging buffer (`ImageCopyDriver` / double-buffered async channel or synchronous frame-locked capture).
- **Output Sinks:**
  - If `<OUTPUT>` ends in `.mp4` / `.mov` / `.mkv`: spawn `ffmpeg -y -f rawvideo -pix_fmt rgba -s WxH -r FPS -i - -c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p <OUTPUT>` with piped `stdin`, streaming raw frame buffers directly without touching disk for intermediates.
  - If `<OUTPUT>` is a directory path: create directory and encode numbered `frame_000001.png` files on a bounded worker thread pool.
- Automatically hold for `2.0s` of video time on the finished mural after the robot parks at `HomePose`, flush `ffmpeg`, and exit cleanly (when `--headless` or upon completion).

---

## 8. Recommended Crate Selection

| Concern | Recommended Crates | Rationale |
| :--- | :--- | :--- |
| **3D Engine & PBR** | `bevy` (latest stable `0.15` / `0.16` — verify via `cargo info bevy`) | First-class PBR, ACES tonemapping, gizmos, dynamic `Image` assets, Linux Wayland/X11/headless support |
| **Linear Algebra (Core)** | `glam` (matches Bevy natively) + `nalgebra` (for `6×7` SVD / DLS Jacobian inversion) | Avoids hand-rolling matrix factorizations while keeping `Vec3`/`Quat` zero-cost compatible with Bevy |
| **SVG & Typography** | `usvg`, `fontdb` | Industry-standard Rust SVG tree normalizer with full system font `<text>`-to-path support |
| **2D Geometry & Offsets** | `geo`, `i_overlay` (or `geo-clipper` / ` cavalier_contours`), `kurbo` | Bulletproof polygon boolean clipping, winding normalization, and polygon inset/offsetting for contour+hatch CAM |
| **CLI** | `clap` (`derive`) | Clean declarative CLI interface |
| **Image Encoding** | `png` / `image` | Fast PNG sequence saving & procedural startup normal/HDRI texture generation |
| **Testing** | `approx` | Floating-point tolerances for FK/IK and geometry unit tests |
