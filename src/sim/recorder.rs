use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Child, ChildStdin, Command, Stdio},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::JoinHandle,
};

use bevy::{
    app::AppExit,
    camera::RenderTarget,
    prelude::*,
    render::{
        Extract, Render, RenderApp, RenderSystems,
        render_asset::RenderAssets,
        render_resource::{
            Buffer, BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, MapMode,
            PollType, TexelCopyBufferInfo, TexelCopyBufferLayout,
        },
        renderer::{RenderContext, RenderDevice, RenderGraph, RenderGraphSystems, RenderQueue},
        texture::GpuImage,
        view::screenshot::{Screenshot, ScreenshotCaptured},
    },
};
use crossbeam_channel::{Receiver, Sender};

use crate::sim::state::{AppConfig, OffscreenRenderTarget, SimClock, SimStepSet};

/// Resource set to `true` when the user presses `Esc`/`Q` so the recorder finalizes cleanly
/// before emitting [`AppExit::Success`].
#[derive(Resource, Debug, Default)]
pub struct GracefulShutdownRequested(pub bool);

/// Registers deterministic offscreen (`ImageCopyDriver`) and windowed (`Screenshot`) capture
/// systems for video (`.mp4`/`.mov`/`.mkv` via `ffmpeg`) and PNG sequence export.
pub fn plugin(app: &mut App) {
    let (sender, receiver) = crossbeam_channel::unbounded::<Vec<u8>>();

    app.insert_resource(MainWorldReceiver(receiver))
        .init_resource::<GracefulShutdownRequested>()
        .init_resource::<WindowedCaptureQueue>()
        .add_systems(Startup, setup_recorder)
        .add_systems(Update, record_capture_system.in_set(SimStepSet::RecordCapture));

    if let Some(render_app) = app.get_sub_app_mut(RenderApp) {
        render_app
            .insert_resource(RenderWorldSender(sender))
            .init_resource::<ImageCopiers>()
            .add_systems(ExtractSchedule, image_copy_extract)
            .add_systems(
                Render,
                receive_image_from_buffer.after(RenderSystems::Render),
            )
            .add_systems(
                RenderGraph,
                image_copy_driver.after(RenderGraphSystems::Submit),
            );
    }
}

/// Receives tightly packed RGBA8 frames from the render world in the main world.
#[derive(Resource, Deref)]
struct MainWorldReceiver(Receiver<Vec<u8>>);

/// Sends tightly packed RGBA8 frames from the render world to the main world.
#[derive(Resource, Deref)]
struct RenderWorldSender(Sender<Vec<u8>>);

/// Aggregated list of active GPU image copiers extracted into the render world.
#[derive(Clone, Default, Resource, Deref, DerefMut)]
struct ImageCopiers(Vec<ImageCopier>);

/// Component attached to an entity in the main world to read back an offscreen render target.
#[derive(Clone, Component)]
struct ImageCopier {
    buffer: Buffer,
    src_image: Handle<Image>,
    width: u32,
    height: u32,
    padded_bytes_per_row: usize,
    enabled: Arc<AtomicBool>,
}

impl ImageCopier {
    fn new(
        src_image: Handle<Image>,
        width: u32,
        height: u32,
        render_device: &RenderDevice,
    ) -> Self {
        let row_bytes = width as usize * 4;
        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(row_bytes);
        let buffer = render_device.create_buffer(&BufferDescriptor {
            label: Some("spraybot_readback_buffer"),
            size: padded_bytes_per_row as u64 * height as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            src_image,
            width,
            height,
            padded_bytes_per_row,
            enabled: Arc::new(AtomicBool::new(true)),
        }
    }

    fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
}

/// Strips `wgpu::COPY_BYTES_PER_ROW_ALIGNMENT` (256-byte) padding from a mapped GPU texture buffer
/// into a tightly packed `width * height * 4` RGBA8 byte buffer.
pub fn unpack_padded_rgba_rows(
    raw: &[u8],
    width: u32,
    height: u32,
    aligned_bytes_per_row: usize,
) -> Vec<u8> {
    let row_bytes = width as usize * 4;
    let h = height as usize;
    let expected_tight = row_bytes * h;

    if row_bytes == 0 || h == 0 {
        return Vec::new();
    }

    if aligned_bytes_per_row == row_bytes && raw.len() >= expected_tight {
        return raw[..expected_tight].to_vec();
    }

    let mut out = Vec::with_capacity(expected_tight);
    for row in raw.chunks(aligned_bytes_per_row).take(h) {
        let take = row_bytes.min(row.len());
        out.extend_from_slice(&row[..take]);
        if take < row_bytes {
            out.resize(out.len() + (row_bytes - take), 0);
        }
    }
    out
}

fn image_copy_extract(mut commands: Commands, copiers: Extract<Query<&ImageCopier>>) {
    commands.insert_resource(ImageCopiers(copiers.iter().cloned().collect()));
}

fn image_copy_driver(
    render_context: RenderContext,
    image_copiers: Res<ImageCopiers>,
    render_queue: Res<RenderQueue>,
    gpu_images: Res<RenderAssets<GpuImage>>,
) {
    for copier in image_copiers.iter() {
        if !copier.enabled() {
            continue;
        }
        let Some(src_image) = gpu_images.get(&copier.src_image) else {
            continue;
        };

        let mut encoder = render_context
            .render_device()
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("spraybot_image_copy_encoder"),
            });

        let block_dimensions = src_image.texture_descriptor.format.block_dimensions();
        let block_size = src_image
            .texture_descriptor
            .format
            .block_copy_size(None)
            .unwrap_or(4);
        let padded_bytes_per_row = RenderDevice::align_copy_bytes_per_row(
            (src_image.texture_descriptor.size.width as usize / block_dimensions.0 as usize)
                * block_size as usize,
        );

        encoder.copy_texture_to_buffer(
            src_image.texture.as_image_copy(),
            TexelCopyBufferInfo {
                buffer: &copier.buffer,
                layout: TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: std::num::NonZero::<u32>::new(padded_bytes_per_row as u32)
                        .map(u32::from),
                    rows_per_image: None,
                },
            },
            Extent3d {
                width: copier.width.min(src_image.texture_descriptor.size.width),
                height: copier.height.min(src_image.texture_descriptor.size.height),
                depth_or_array_layers: 1,
            },
        );

        render_queue.submit(std::iter::once(encoder.finish()));
    }
}

fn receive_image_from_buffer(
    image_copiers: Res<ImageCopiers>,
    render_device: Res<RenderDevice>,
    sender: Res<RenderWorldSender>,
) {
    for copier in image_copiers.iter() {
        if !copier.enabled() {
            continue;
        }

        let buffer_slice = copier.buffer.slice(..);
        let (tx, rx) = crossbeam_channel::bounded(1);

        buffer_slice.map_async(MapMode::Read, move |res| {
            let _ = tx.send(res);
        });

        render_device
            .poll(PollType::wait_indefinitely())
            .expect("failed to poll GPU device for readback map_async");

        match rx.recv() {
            Ok(Ok(())) => {
                let mapped = buffer_slice
                    .get_mapped_range()
                    .expect("failed to get mapped readback buffer range");
                let tight = unpack_padded_rgba_rows(
                    &mapped,
                    copier.width,
                    copier.height,
                    copier.padded_bytes_per_row,
                );
                drop(mapped);
                copier.buffer.unmap();
                let _ = sender.send(tight);
            }
            Ok(Err(err)) => {
                error!("GPU readback buffer map_async failed: {err}");
            }
            Err(err) => {
                error!("GPU readback channel recv failed: {err}");
            }
        }
    }
}

/// Identifies whether an output path should be piped to `ffmpeg` or written as a PNG sequence.
pub fn is_video_file_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "mp4" | "mov" | "mkv"
            )
        })
}

/// Destination sink for tightly packed RGBA8 frames.
enum VideoSink {
    Ffmpeg {
        output_path: PathBuf,
        child: Child,
        stdin: Option<ChildStdin>,
        stderr_thread: Option<JoinHandle<String>>,
        frames_written: u64,
    },
    PngSequence {
        output_dir: PathBuf,
        tx: Option<Sender<(u64, Vec<u8>)>>,
        workers: Vec<JoinHandle<()>>,
        frames_written: u64,
    },
}

impl VideoSink {
    fn open(output: &Path, width: u32, height: u32, fps: u32) -> Result<Self, String> {
        if is_video_file_path(output) {
            if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
                fs::create_dir_all(parent).map_err(|err| {
                    format!(
                        "failed to create parent directory '{}': {err}",
                        parent.display()
                    )
                })?;
            }

            let mut child = Command::new("ffmpeg")
                .args([
                    "-y",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgba",
                    "-s",
                    &format!("{width}x{height}"),
                    "-r",
                    &fps.to_string(),
                    "-i",
                    "-",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                ])
                .arg(output)
                .stdin(Stdio::piped())
                .stdout(Stdio::null())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|err| format!("failed to spawn ffmpeg for '{}': {err}", output.display()))?;

            let stdin = child
                .stdin
                .take()
                .ok_or_else(|| "ffmpeg spawned without stdin pipe".to_string())?;
            let stderr = child.stderr.take();
            let stderr_thread = stderr.map(|mut pipe| {
                std::thread::spawn(move || {
                    let mut buf = String::new();
                    let _ = pipe.read_to_string(&mut buf);
                    buf
                })
            });

            info!(
                "Recording {width}x{height} @ {fps}fps MP4/video stream to {}",
                output.display()
            );

            Ok(Self::Ffmpeg {
                output_path: output.to_path_buf(),
                child,
                stdin: Some(stdin),
                stderr_thread,
                frames_written: 0,
            })
        } else {
            fs::create_dir_all(output).map_err(|err| {
                format!(
                    "failed to create PNG output directory '{}': {err}",
                    output.display()
                )
            })?;

            let num_workers = std::thread::available_parallelism()
                .map(|n| n.get().clamp(2, 8))
                .unwrap_or(4);
            let (tx, rx) = crossbeam_channel::bounded::<(u64, Vec<u8>)>(16);
            let mut workers = Vec::with_capacity(num_workers);

            for _ in 0..num_workers {
                let rx_worker = rx.clone();
                let dir = output.to_path_buf();
                workers.push(std::thread::spawn(move || {
                    while let Ok((frame_idx, rgba)) = rx_worker.recv() {
                        let file_path = dir.join(format!("frame_{frame_idx:06}.png"));
                        if let Err(err) = write_rgba_png(&file_path, width, height, &rgba) {
                            error!("Failed to write '{}': {err}", file_path.display());
                        }
                    }
                }));
            }

            info!(
                "Recording {width}x{height} @ {fps}fps PNG sequence into {}/ ({num_workers} workers)",
                output.display()
            );

            Ok(Self::PngSequence {
                output_dir: output.to_path_buf(),
                tx: Some(tx),
                workers,
                frames_written: 0,
            })
        }
    }

    fn write_frame(&mut self, rgba: Vec<u8>) -> Result<(), String> {
        match self {
            Self::Ffmpeg {
                stdin,
                frames_written,
                ..
            } => {
                let Some(pipe) = stdin.as_mut() else {
                    return Err("ffmpeg stdin already closed".to_string());
                };
                pipe.write_all(&rgba)
                    .map_err(|err| format!("failed to pipe frame to ffmpeg: {err}"))?;
                *frames_written += 1;
                Ok(())
            }
            Self::PngSequence {
                tx,
                frames_written,
                ..
            } => {
                let Some(sender) = tx.as_ref() else {
                    return Err("PNG worker channel already closed".to_string());
                };
                *frames_written += 1;
                let frame_idx = *frames_written;
                sender
                    .send((frame_idx, rgba))
                    .map_err(|err| format!("failed to queue PNG frame {frame_idx}: {err}"))
            }
        }
    }

    fn finalize(&mut self) -> Result<u64, String> {
        match self {
            Self::Ffmpeg {
                output_path,
                child,
                stdin,
                stderr_thread,
                frames_written,
            } => {
                if let Some(mut pipe) = stdin.take() {
                    let _ = pipe.flush();
                    drop(pipe);
                }
                let status = child
                    .wait()
                    .map_err(|err| format!("failed waiting for ffmpeg exit: {err}"))?;
                let stderr_text = stderr_thread
                    .take()
                    .and_then(|h| h.join().ok())
                    .unwrap_or_default();

                if !status.success() {
                    return Err(format!(
                        "ffmpeg exited with non-zero status {status} for '{}': {}",
                        output_path.display(),
                        stderr_text.trim()
                    ));
                }

                info!(
                    "Saved {} frames to {}",
                    *frames_written,
                    output_path.display()
                );
                Ok(*frames_written)
            }
            Self::PngSequence {
                output_dir,
                tx,
                workers,
                frames_written,
            } => {
                drop(tx.take());
                for handle in workers.drain(..) {
                    if handle.join().is_err() {
                        error!("A PNG encoder worker panicked");
                    }
                }
                info!(
                    "Saved {} PNG frames into {}",
                    *frames_written,
                    output_dir.display()
                );
                Ok(*frames_written)
            }
        }
    }
}

impl Drop for VideoSink {
    fn drop(&mut self) {
        match self {
            Self::Ffmpeg {
                stdin,
                child,
                stderr_thread,
                ..
            } => {
                if stdin.is_some() {
                    drop(stdin.take());
                    let _ = child.wait();
                }
                if let Some(h) = stderr_thread.take() {
                    let _ = h.join();
                }
            }
            Self::PngSequence { tx, workers, .. } => {
                if tx.is_some() {
                    drop(tx.take());
                }
                for handle in workers.drain(..) {
                    let _ = handle.join();
                }
            }
        }
    }
}

/// Writes a tightly packed RGBA8 buffer as a PNG file using the `png` crate.
pub fn write_rgba_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<(), String> {
    let expected = width as usize * height as usize * 4;
    if rgba.len() != expected {
        return Err(format!(
            "RGBA buffer length mismatch: expected {expected}, got {}",
            rgba.len()
        ));
    }

    let file =
        fs::File::create(path).map_err(|err| format!("failed to create '{}': {err}", path.display()))?;
    let writer = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(writer, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.set_compression(png::Compression::Fast);
    let mut png_writer = encoder
        .write_header()
        .map_err(|err| format!("failed to write PNG header: {err}"))?;
    png_writer
        .write_image_data(rgba)
        .map_err(|err| format!("failed to write PNG image data: {err}"))?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaptureMode {
    Offscreen,
    Windowed,
}

#[derive(Resource)]
struct RecorderState {
    mode: CaptureMode,
    sink: VideoSink,
    warmup_drained: bool,
    finishing: bool,
    flush_ticks_remaining: u32,
    finalized: bool,
    next_capture_seq: u64,
    next_write_seq: u64,
}

#[derive(Component, Debug, Clone, Copy)]
struct CaptureFrameSeq(u64);

#[derive(Resource, Default)]
struct WindowedCaptureQueue {
    frames: BTreeMap<u64, Vec<u8>>,
}

fn setup_recorder(
    mut commands: Commands,
    app_config: Res<AppConfig>,
    offscreen: Res<OffscreenRenderTarget>,
    render_device: Res<RenderDevice>,
    mut app_exit: MessageWriter<AppExit>,
) {
    let Some(output_path) = app_config.cli.record.as_ref() else {
        return;
    };

    let (width, height) = app_config.cli.resolution;
    let fps = app_config.cli.fps.max(1);

    let sink = match VideoSink::open(output_path, width, height, fps) {
        Ok(sink) => sink,
        Err(err) => {
            error!("Recorder initialization failed: {err}");
            app_exit.write(AppExit::error());
            return;
        }
    };

    let mode = if let Some(image_handle) = offscreen.image.as_ref() {
        commands.spawn(ImageCopier::new(
            image_handle.clone(),
            width,
            height,
            &render_device,
        ));
        CaptureMode::Offscreen
    } else {
        CaptureMode::Windowed
    };

    commands.insert_resource(RecorderState {
        mode,
        sink,
        warmup_drained: false,
        finishing: false,
        flush_ticks_remaining: 0,
        finalized: false,
        next_capture_seq: 0,
        next_write_seq: 0,
    });
}

fn on_screenshot_captured(
    captured: On<ScreenshotCaptured>,
    seq_query: Query<&CaptureFrameSeq>,
    mut queue: ResMut<WindowedCaptureQueue>,
    app_config: Res<AppConfig>,
) {
    let Ok(seq) = seq_query.get(captured.entity) else {
        return;
    };

    let (target_w, target_h) = app_config.cli.resolution;
    let img = captured.image.clone();
    let rgba_bytes = match img.try_into_dynamic() {
        Ok(dyn_img) => {
            let rgba = if dyn_img.width() != target_w || dyn_img.height() != target_h {
                dyn_img
                    .resize_exact(target_w, target_h, image::imageops::FilterType::Triangle)
                    .to_rgba8()
            } else {
                dyn_img.to_rgba8()
            };
            rgba.into_raw()
        }
        Err(err) => {
            error!("Failed to convert windowed screenshot to RGBA8: {err:?}");
            return;
        }
    };

    queue.frames.insert(seq.0, rgba_bytes);
}

#[allow(clippy::too_many_arguments)]
fn record_capture_system(
    mut commands: Commands,
    recorder: Option<ResMut<RecorderState>>,
    clock: Res<SimClock>,
    shutdown_req: Res<GracefulShutdownRequested>,
    receiver: Res<MainWorldReceiver>,
    copiers: Query<&ImageCopier>,
    mut windowed_queue: ResMut<WindowedCaptureQueue>,
    mut app_exit: MessageWriter<AppExit>,
) {
    let Some(mut state) = recorder else {
        return;
    };
    if state.finalized {
        return;
    }

    let stop_requested = clock.finished || shutdown_req.0;

    match state.mode {
        CaptureMode::Offscreen => {
            if clock.warmup_frames_remaining > 0 && !stop_requested {
                // Discard frames rendered before warmup finishes.
                while receiver.try_recv().is_ok() {}
                return;
            }

            if !state.warmup_drained {
                // On the first tick where warmup reaches 0, the channel still holds the frame
                // rendered at the end of the previous warmup tick; discard it once.
                while receiver.try_recv().is_ok() {}
                state.warmup_drained = true;
                if !stop_requested {
                    return;
                }
            }

            while let Ok(frame_rgba) = receiver.try_recv() {
                if let Err(err) = state.sink.write_frame(frame_rgba) {
                    error!("{err}");
                    finalize_and_exit(&mut state, &copiers, &mut app_exit);
                    return;
                }
            }

            if stop_requested {
                if !state.finishing {
                    state.finishing = true;
                    // Allow one final tick so the render submitted at the end of this frame
                    // is read back and written before closing ffmpeg.
                    state.flush_ticks_remaining = 1;
                } else if state.flush_ticks_remaining > 0 {
                    state.flush_ticks_remaining -= 1;
                    if state.flush_ticks_remaining == 0 {
                        for copier in copiers.iter() {
                            copier.set_enabled(false);
                        }
                        while let Ok(frame_rgba) = receiver.try_recv() {
                            if let Err(err) = state.sink.write_frame(frame_rgba) {
                                error!("{err}");
                                break;
                            }
                        }
                        finalize_and_exit(&mut state, &copiers, &mut app_exit);
                    }
                } else {
                    finalize_and_exit(&mut state, &copiers, &mut app_exit);
                }
            }
        }
        CaptureMode::Windowed => {
            if clock.warmup_frames_remaining > 0 && !stop_requested {
                return;
            }

            while let Some(rgba) = windowed_queue.frames.remove(&state.next_write_seq) {
                if let Err(err) = state.sink.write_frame(rgba) {
                    error!("{err}");
                    finalize_and_exit(&mut state, &copiers, &mut app_exit);
                    return;
                }
                state.next_write_seq += 1;
            }

            if !stop_requested {
                let seq = state.next_capture_seq;
                state.next_capture_seq += 1;
                commands
                    .spawn((Screenshot(RenderTarget::default()), CaptureFrameSeq(seq)))
                    .observe(on_screenshot_captured);
            } else {
                if !state.finishing {
                    state.finishing = true;
                    state.flush_ticks_remaining = 30;
                } else {
                    state.flush_ticks_remaining = state.flush_ticks_remaining.saturating_sub(1);
                }

                if state.next_write_seq >= state.next_capture_seq || state.flush_ticks_remaining == 0
                {
                    finalize_and_exit(&mut state, &copiers, &mut app_exit);
                }
            }
        }
    }
}

fn finalize_and_exit(
    state: &mut RecorderState,
    copiers: &Query<&ImageCopier>,
    app_exit: &mut MessageWriter<AppExit>,
) {
    if state.finalized {
        return;
    }
    for copier in copiers.iter() {
        copier.set_enabled(false);
    }
    state.finalized = true;
    match state.sink.finalize() {
        Ok(_) => {
            app_exit.write(AppExit::Success);
        }
        Err(err) => {
            error!("Recorder finalization error: {err}");
            app_exit.write(AppExit::error());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_padded_rgba_rows_strips_256_byte_alignment() {
        // 100x3 image -> tight row = 400 bytes, aligned row = 512 bytes.
        let width = 100;
        let height = 3;
        let tight_row = width as usize * 4;
        let aligned_row = 512usize;
        let mut raw = vec![0xAAu8; aligned_row * height as usize];

        for y in 0..height as usize {
            for x in 0..tight_row {
                raw[y * aligned_row + x] = ((y * 37 + x) % 251) as u8;
            }
        }

        let unpacked = unpack_padded_rgba_rows(&raw, width, height, aligned_row);
        assert_eq!(unpacked.len(), tight_row * height as usize);
        for y in 0..height as usize {
            for x in 0..tight_row {
                assert_eq!(
                    unpacked[y * tight_row + x],
                    ((y * 37 + x) % 251) as u8,
                    "mismatch at row {y} byte {x}"
                );
            }
        }
    }

    #[test]
    fn unpack_padded_rgba_rows_handles_exact_alignment() {
        // 64x2 image -> tight row = 256 bytes, aligned row = 256 bytes.
        let raw: Vec<u8> = (0..(64 * 2 * 4)).map(|i| (i % 256) as u8).collect();
        let unpacked = unpack_padded_rgba_rows(&raw, 64, 2, 256);
        assert_eq!(unpacked, raw);
    }

    #[test]
    fn detects_video_extensions_vs_png_directories() {
        assert!(is_video_file_path(Path::new("/tmp/out.mp4")));
        assert!(is_video_file_path(Path::new("/tmp/out.MP4")));
        assert!(is_video_file_path(Path::new("/tmp/out.mov")));
        assert!(is_video_file_path(Path::new("/tmp/out.mkv")));
        assert!(!is_video_file_path(Path::new("/tmp/frames_dir")));
        assert!(!is_video_file_path(Path::new("/tmp/frames_dir/")));
    }

    #[test]
    fn write_rgba_png_creates_valid_png_file() {
        let dir = std::env::temp_dir().join("spraybot_recorder_png_test");
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("frame_000001.png");

        let rgba = vec![255u8; 4 * 4 * 4];
        write_rgba_png(&path, 4, 4, &rgba).expect("write valid PNG");
        let bytes = fs::read(&path).expect("read PNG back");
        assert_eq!(&bytes[..8], b"\x89PNG\r\n\x1a\n");

        let _ = fs::remove_dir_all(&dir);
    }
}
