# SprayBot — Implementation Plan (Orchestrator Guide)

This plan breaks the implementation specified in [DESIGN.md](./DESIGN.md) into modular, verifiable phases with explicit interfaces so subsystems can be built and tested in parallel without merge conflicts.

---

## Architectural Contract Strategy (Read First)

Before parallelizing work across modules, establish thin shared data types in `src/core/types.rs` (or `src/core/mod.rs`) so all pure-Rust planning modules (`svg_ingest`, `palette`, `batching`, `planner`, `kinematics`, `trajectory`) compile and test in milliseconds **without linking Bevy**:

1. **`NormalizedPrimitive`**:
   - `id: usize` (SVG painter's Z-order)
   - `color_srgb: [u8; 3]`, `opacity: f32`
   - `geometry: PrimitiveGeom` (`FilledPolygon(geo::MultiPolygon<f64>)` | `StrokedPath { points: Vec<DVec2>, width: f64 }`)
2. **`PalettePlan`**:
   - `cans: Vec<SprayCanSpec>` (`slot_index`, `color_srgb`, `label`)
   - `primitive_color_map: Vec<usize>` (maps `primitive_id -> slot_index`)
3. **`SprayOperation` / `JobPlan`**:
   - Ordered sequence of `JobOp`:
     - `PickCan { slot: usize }`
     - `ShakeCan`
     - `PaintStroke { points_canvas_3d: Vec<StrokeWaypoint> }` where `StrokeWaypoint` carries `pos: Vec3`, `normal: Vec3`, `standoff: f32`, `feed_speed: f32`, `valve_on: bool`, `opacity_scale: f32`
     - `ReturnCan { slot: usize }`
     - `ParkHome`
4. **`KinematicSample` / `CompiledTimeline`**:
   - Continuous/interpolable trajectory `sample(t_sim: f64) -> RobotFrameState`:
     - `joints: [f64; 7]` (`q[0]` = rail X in meters, `q[1..7]` = arm joint angles in radians)
     - `gripper_open: f32` (`0.0` closed on can .. `1.0` open)
     - `trigger_depressed: f32` (`0.0` off .. `1.0` spraying)
     - `held_can_slot: Option<usize>`
     - `nozzle_world_pos: Vec3`, `nozzle_world_dir: Vec3`, `spray_active: bool`, `flow_scale: f32`

---

## Phase 1: Workspace Bootstrap & Shared Core Contracts
**Goal:** Working Cargo project, verified crate compatibility on Fedora Linux (`rustc 1.99+`), CLI skeleton, and shared core types + sample SVGs.

- [ ] **1.1 Initialize `Cargo.toml`**:
  - Check latest compatible `bevy` version (`cargo search bevy` / `cargo info bevy`), enable `dynamic_linking` only behind an optional `--features fast-dev` flag so release builds stay standalone.
  - Add core dependencies: `clap` (`derive`), `glam`, `nalgebra`, `usvg`, `fontdb`, `kurbo`, `geo`, `i_overlay` (or `geo-buf` / polygon offset crate — verify API via `cargo check`), `image`, `png`, `approx`.
- [ ] **1.2 Author `src/cli.rs` & `src/core/types.rs`**:
  - Implement exact CLI options from `DESIGN.md §7.1`.
  - Define the shared core contracts above.
- [ ] **1.3 Create Built-In Test SVGs (`assets/examples/`)**:
  - `01_geometric_shapes.svg`: Crisp strokes of varying `stroke-width`, overlapping circles/stars, and compound polygon with holes (`evenodd`).
  - `02_street_art_poster.svg`: Multi-layered mural with `<text>` typography, semi-transparent shadow layers, and a `<linearGradient>` background sky.
  - `03_palette_stress.svg`: 25+ color mosaic to verify deterministic OKLab k-means quantization down to `--max-cans 16`.
- **Verification Gate 1:** `cargo check` and `cargo test` pass cleanly.

---

## Phase 2: Pure-Rust Core Engines (Parallelizable!)
*Note for Orchestrator: Tracks 2A and 2B touch completely separate files under `src/core/` and can be built simultaneously by independent workers.*

### Track 2A: SVG → Palette → Batching → 2D/3D CAM Planner
- [ ] **2A.1 `svg_ingest.rs`**:
  - Load SVG via `usvg::Tree` with system fonts (`fontdb::Database::load_system_fonts()`).
  - Compute uniform aspect-preserving transform into the `2.50m × 1.50m` wall canvas (`Y ∈ [0.80, 2.30]`, `X ∈ [-1.25, +1.25]`, margin `0.08m`).
  - Flatten Bézier curves (`tolerance <= 0.5mm`), normalize `evenodd`/`nonzero` compound paths into valid `geo::MultiPolygon<f64>`, extract strokes, and slice `<linearGradient>`/`<radialGradient>` fills into `4–6` stepped color band polygons (with average-color fallback).
- [ ] **2A.2 `palette.rs`**:
  - Convert colors to OKLab, deduplicate within `ΔE < 0.015`, and implement deterministic seeded area-weighted k-means++ when unique colors exceed `max_cans`.
- [ ] **2A.3 `batching.rs`**:
  - Build back-to-front overlap DAG (expanding footprints by `8mm` spray feather margin) and schedule greedy topological color batches + nearest-neighbor intra-batch stroke ordering to minimize rapid air moves and rack swaps.
- [ ] **2A.4 `planner.rs`**:
  - Implement nozzle cone mapping `w(d) = 2 * d * tan(12°)` for `d ∈ [0.04m, 0.26m]`.
  - Generate single-pass dynamic-standoff strokes (or buffered polygon fills for extra-wide strokes).
  - Generate contour-then-hatch polygon fills: inset crisp perimeter pass (`d = 0.05m`) + boustrophedon interior zig-zag scanlines (`35%` overlap step-over, thin-sliver centerline fallback).
  - Add tangential lead-in / lead-out segments (`3.5cm`, `valve_on = false`) and standoff-compensated `feed_speed`.
- **Unit Tests (2A):** Test SVG aspect scaling, `<text>` outline extraction, hole preservation, k-means cap (`25 -> 16` colors), Z-order preservation on overlapping shapes vs batching on disjoint shapes, and zero-valve lead-in/lead-out invariants.

### Track 2B: 7-DOF Kinematics, Weighted DLS IK & Trajectory Compiler
- [ ] **2B.1 `kinematics.rs`**:
  - Implement data-driven `KinematicChain` (`1 Prismatic` floor rail + `6 Revolute` industrial arm links, reach `~1.35m`, TCP offset to held aerosol nozzle).
  - Implement Forward Kinematics (`fk_all_links` + `fk_nozzle`) and analytical geometric Jacobian `J(q) ∈ ℝ^{6×7}`.
  - Implement Weighted Damped Least-Squares IK solver (`nalgebra` `6×6` damped Cholesky/LU solve) with:
    - Heavy rail weight (`w_0 = 6.0`) vs medium arm (`1.5`) vs light wrist (`0.4`).
    - Axial spray-cone roll relaxation (soft upright `+Y` bias during painting; full 6-DOF pose lock during rack pick/place).
    - Null-space posture bias toward elbow-up `q_comfort` and joint-limit repulsion.
- [ ] **2B.2 `trajectory.rs`**:
  - Compile `JobPlan` into a continuous time-parameterized `CompiledTimeline` respecting joint velocity/acceleration limits (trapezoidal / S-curve profiles).
  - Implement full rack pick/place choreography: safe approach pose → linear slot insert → gripper close → vertical lift → retract → `1.1s` sinusoidal wrist-shake → safe Z-retract transits → steady-speed spray passes → return can → park home.
- **Unit Tests (2B):**
  - Verify numerical finite-difference Jacobian matches analytical `J(q)` to `< 1e-5`.
  - Sweep a grid of `(X, Y, Z)` targets across all 4 corners + center of the `2.5m × 1.5m` canvas and all `16` rack slots, asserting IK position error `< 0.5mm`, pointing error `< 0.2°`, and zero joint-limit violations!

---

## Phase 3: Bevy 3D Studio, Procedural Robot & PBR Canvas
**Goal:** Interactive photorealistic 3D scene driving the robot along `CompiledTimeline` and painting dynamically onto the wall canvas.

- [ ] **3.1 `environment.rs` (Industrial Mural Studio)**:
  - Build concrete warehouse floor, textured back wall, baseboards/pillars, steel floor rail track (`3.8m`), and 2-tier 16-slot angled spray can rack.
  - Generate procedural equirectangular studio reflection map (`EnvironmentMapLight`) + shadowed directional skylight + warehouse spotlights + ACES tonemapping & subtle bloom.
- [ ] **3.2 `robot_view.rs` & `controller.rs`**:
  - Spawn hierarchical/link-indexed procedural PBR robot meshes (graphite/cast-iron carriage, industrial orange/charcoal arm links, machined aluminum wrist, moving parallel gripper jaws, and actuating nozzle trigger finger).
  - Spawn aerosol cans in rack slots colored to `PalettePlan` (metallic rim/dome + colored cap & body band).
  - Drive link transforms, gripper jaws, trigger finger, and held-can attachment each tick from `CompiledTimeline::sample(t_sim)`.
- [ ] **3.3 `canvas.rs` & `mist.rs` (PBR Spray Deposition & FX)**:
  - Generate startup procedural linen weave normal map + warm gesso albedo + matte roughness (`0.88`) texture.
  - Implement sub-stepped ray-plane cone deposition kernel (`Δs <= 0.25 * R`) updating both `albedo_tex` (Gaussian core + outer hash-noise overspray speckle) and `metallic_roughness_tex` (transitioning painted strokes to `0.32` satin roughness) and uploading changes to GPU `Assets<Image>`.
  - Add billboard/instanced cone mist puffs active during `spray_active == true`.
- [ ] **3.4 `camera.rs` (Controls & Gizmos)**:
  - Smooth damped mouse orbit (`Left-drag`), pan (`Right/Middle-drag`), scroll zoom, plus hotkeys `1–4` (Overview, Action Follow, Rack Closeup, Canvas Front), `Space` (Pause), `[`/`]` (Speed), `G` (Gizmo toolpath/IK overlay), `R` (Restart).

---

## Phase 4: Deterministic Video Export & End-to-End Verification
**Goal:** Frame-accurate offline MP4 / PNG sequence rendering and visual polish check.

- [ ] **4.1 `recorder.rs` (Fixed-Timestep GPU Readback & `ffmpeg` Pipe)**:
  - Lock simulation step `Δt_sim = speed / fps` per rendered frame when `--record` is active.
  - Configure offscreen render target / `ImageCopyDriver` GPU-to-CPU frame readback (supporting both windowed + `--record` and `--headless --record`).
  - Pipe raw `RGBA8` frames into `ffmpeg` stdin (`libx264 -crf 18 -pix_fmt yuv420p`) when output is `.mp4`/`.mkv`, or write numbered PNGs when output is a directory.
  - Include `2.0s` end-card hold after `ParkHome` before cleanly closing `ffmpeg` stdin and awaiting exit code `0`.
- [ ] **4.2 Visual & Headless Integration Verification**:
  - Run headless render test:
    ```bash
    cargo run --release -- assets/examples/02_street_art_poster.svg --speed 15 --fps 30 --resolution 1280x720 --headless --record /tmp/spraybot_verify.mp4
    ```
  - Verify output video duration/integrity with `ffprobe` and extract key frames (rack pickup/shake, mid-painting mist + satin reflection, final completed mural) to visually inspect lighting, arm posture, and crisp SVG reproduction!

---

## Known Pitfalls & Guardrails for Implementers
1. **Bevy Headless Rendering on Linux:** When `--headless` is set, disable `WinitPlugin` (use `ScheduleRunnerPlugin` driving the render schedule) *while keeping* `RenderPlugin` active on an offscreen `Image` render target—do **not** use `MinimalPlugins` alone or the GPU PBR renderer won't run.
2. **High-Speed Timelapse Gaps (`--speed 20`):** Never splat just once per rendered frame! Always integrate/sub-step along `CompiledTimeline` between `t_prev` and `t_curr` at spatial intervals `<= 0.25 * spot_radius` so fast timelapse videos still lay down continuous, silky strokes.
3. **sRGB vs Linear Color Math:** Bevy's `Rgba8UnormSrgb` decodes to linear on GPU automatically; keep CPU paint blending consistent (convert sRGB palette colors to linear float `[f32; 4]` in a CPU backing buffer, blend in linear space, encode back to sRGB bytes on dirty-rect or full texture upload).
4. **Polygon Offset Robustness:** Degenerate zero-area slivers from SVG boolean/inset operations should be filtered (`area < 1e-6 m²`) and tiny thin features narrower than `2 * r_inset` should fall back cleanly to a single close-standoff centerline pass so small SVG text characters never vanish!
