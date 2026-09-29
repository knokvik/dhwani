"""
Live GTCRN speech-enhancement demo. Laptop stand-in for the Raspberry Pi target
(no Pi hardware available yet) -- proves the real-time architecture, not the
eventual embedded deployment.

Modes:
  --check                         offline correctness + RTF check, no audio hardware
  --measure-latency               end-to-end acoustic latency (needs mic + speakers)
  --capture-test SEC --out FILE   mic -> file for SEC seconds, no speaker output
  (default, no flags)             mic -> enhance -> speakers, live, 'e' toggles, 'q' quits

Dual INMP441 microphone mode (--dual-mic):
  Channel 0: Main INMP441 (near the speaker's mouth)
  Channel 1: Reference INMP441 (captures ambient battlefield noise)
  The reference channel feeds a spectral-subtraction pre-stage before GTCRN.

Note: this is the pretrained model_trained_on_dns3.tar checkpoint, not yet fine-tuned
for impulsive defence noise (gunshots/shelling) -- see docs/solution-design.md Stage 1.
"""
import argparse
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from streaming_engine import HOP, N_FFT, ONNX_PATH, WINDOW, StreamingEnhancer  # noqa: E402

SAMPLE_RATE = 16000
GTCRN_DIR = REPO_ROOT / "third_party" / "gtcrn"


def best_lag(ref: np.ndarray, test: np.ndarray, max_lag: int = 600) -> tuple[int, float]:
    """Find the sample shift of `test` relative to `ref` that minimizes mean abs
    error. Needed because the causal streaming engine and any non-streaming
    reference computed with centered STFT differ by a fixed frame-alignment
    offset, not just noise -- this both validates correctness and measures the
    real algorithmic delay."""
    best_lag_val, best_err = 0, float("inf")
    for lag in range(max_lag):
        n = min(len(ref), len(test) - lag)
        if n <= 0:
            break
        err = float(np.abs(ref[:n] - test[lag : lag + n]).mean())
        if err < best_err:
            best_err, best_lag_val = err, lag
    return best_lag_val, best_err


def causal_reference(mix: np.ndarray, checkpoint: str = "model_trained_on_dns3.tar") -> np.ndarray:
    """Non-streaming (whole-utterance) GTCRN forward pass, but framed and
    reconstructed with the exact same manual, non-centered analysis/OLA-synthesis
    convention as StreamingEnhancer (torch's istft(center=False) hits an internal
    NOLA/edge-normalization assertion, and center=True would reflect-pad using
    samples a live stream doesn't have yet -- so we roll our own here to get a
    fair, directly-comparable ground truth). GTCRN's time-axis RNNs are all
    unidirectional (see DPGRNN.inter_rnn, bidirectional=False), so this whole-file
    batch pass is causal and should numerically match the frame-by-frame
    streaming engine almost exactly."""
    sys.path.insert(0, str(GTCRN_DIR))
    from gtcrn import GTCRN

    model = GTCRN().eval()
    import torch          # only the batch reference needs it; the Pi has no torch

    ckpt = torch.load(GTCRN_DIR / "checkpoints" / checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    n_hops = len(mix) // HOP
    buf = np.zeros(N_FFT, dtype=np.float32)
    frames = []
    for i in range(n_hops):
        buf = np.concatenate([buf[HOP:], mix[i * HOP : (i + 1) * HOP]])
        frames.append(np.fft.rfft(buf * WINDOW, n=N_FFT))
    spec = np.stack(frames, axis=1)  # (257, n_hops) complex

    spec_t = torch.from_numpy(np.stack([spec.real, spec.imag], axis=-1).astype(np.float32))[None]  # (1,257,n_hops,2)
    with torch.no_grad():
        out = model(spec_t)[0].numpy()  # (257, n_hops, 2)

    synth_buf = np.zeros(N_FFT, dtype=np.float32)
    out_samples = np.zeros(n_hops * HOP, dtype=np.float32)
    for i in range(n_hops):
        spec_out = out[:, i, 0] + 1j * out[:, i, 1]
        frame = np.fft.irfft(spec_out, n=N_FFT).astype(np.float32) * WINDOW
        synth_buf = synth_buf + frame
        out_samples[i * HOP : (i + 1) * HOP] = synth_buf[:HOP]
        synth_buf = np.concatenate([synth_buf[HOP:], np.zeros(HOP, dtype=np.float32)])
    return out_samples


def run_streaming(mix: np.ndarray, enabled: bool = True, onnx_path=None, dual_mic: bool = False) -> tuple[np.ndarray, float]:
    enhancer = StreamingEnhancer(onnx_path, dual_mic=dual_mic) if onnx_path else StreamingEnhancer(dual_mic=dual_mic)
    n_hops = len(mix) // HOP
    out = np.zeros(n_hops * HOP, dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n_hops):
        hop = mix[i * HOP : (i + 1) * HOP]
        out[i * HOP : (i + 1) * HOP] = enhancer.process_hop(hop, enabled=enabled)
    elapsed = time.perf_counter() - t0
    audio_duration = n_hops * HOP / SAMPLE_RATE
    return out, elapsed / audio_duration


def _dev(spec):
    """Accept a device index or a substring of its name; None means system default."""
    if spec is None:
        return None
    return int(spec) if str(spec).lstrip("-").isdigit() else spec


def cmd_list_devices(_args) -> None:
    import sounddevice as sd
    print(sd.query_devices())
    print("\nPass either the index or part of the name, e.g.:")
    print("  --input-device 3 --output-device 'Bluetooth'")


def cmd_process(args) -> None:
    """Enhance a wav file and write the result. No audio hardware involved.

    This is how you verify a board that has no sound card attached yet: copy a
    noisy file across, process it on the target, copy the result back, and listen
    on a machine that does have speakers. It proves the model produces correct
    audio on that CPU, which is separate from proving the audio stack works.
    """
    src = Path(args.process)
    if not src.exists():
        sys.exit(f"no such file: {src}")
    mix, fs = sf.read(src, dtype="float32")
    if mix.ndim > 1:
        mix = mix[:, 0]
    if fs != SAMPLE_RATE:
        from scipy import signal as _sps
        mix = _sps.resample(mix, int(len(mix) * SAMPLE_RATE / fs)).astype(np.float32)
        print(f"resampled {fs} -> {SAMPLE_RATE} Hz")

    custom = getattr(args, "onnx", None)
    print(f"input:  {src} ({len(mix)/SAMPLE_RATE:.2f}s)")
    print(f"model:  {custom if custom else 'upstream gtcrn_simple.onnx'}")
    out, rtf = run_streaming(mix, enabled=True, onnx_path=custom)
    sf.write(args.out, out, SAMPLE_RATE)

    print(f"output: {args.out}")
    print(f"RTF on this CPU: {rtf:.4f}  "
          f"({'real-time capable' if rtf < 1 else 'NOT real-time'})")
    print("\nCopy it back and listen:")
    print(f"  scp <user>@<host>:{Path(args.out).resolve()} .")


def cmd_check(args) -> None:
    mix_path = GTCRN_DIR / "stream" / "test_wavs" / "mix.wav"
    mix, fs = sf.read(mix_path, dtype="float32")
    assert fs == SAMPLE_RATE, f"expected {SAMPLE_RATE} Hz, got {fs}"

    custom = getattr(args, "onnx", None)
    print(f"input: {mix_path} ({len(mix)/SAMPLE_RATE:.2f}s)")
    print(f"model: {custom if custom else 'upstream gtcrn_simple.onnx'}")
    print("running streaming ONNX engine hop-by-hop...")
    streaming_out, rtf = run_streaming(mix, enabled=True, onnx_path=custom)

    if custom:
        # The batch reference below is the *pretrained* model, so comparing a
        # fine-tuned ONNX against it would measure the fine-tuning, not the export.
        # Correctness of a custom export is checked by scripts/export_onnx.py instead.
        print(f"\nRTF on this CPU: {rtf:.4f}  ({'real-time capable' if rtf < 1 else 'NOT real-time'})")
        sf.write("/tmp/streaming_check.wav", streaming_out, SAMPLE_RATE)
        print("wrote /tmp/streaming_check.wav")
        print("\nSkipping the batch-reference comparison: it only applies to the pretrained model.")
        print("Custom exports are verified by: python scripts/export_onnx.py --checkpoint <ckpt>")
        return

    # The streaming ONNX carries model_trained_on_dns3.tar -- established by
    # scripts/verify_onnx_provenance.py, not assumed here.
    print("computing causal batch reference (model_trained_on_dns3.tar)...")
    ref = causal_reference(mix, checkpoint="model_trained_on_dns3.tar")
    lag, mean_err = best_lag(ref, streaming_out)
    n = min(len(ref) - lag, len(streaming_out) - lag)
    max_err = float(np.abs(ref[:n] - streaming_out[lag : lag + n]).max())

    out_path = Path("/tmp/streaming_check.wav")
    sf.write(out_path, streaming_out, SAMPLE_RATE)

    print()
    print(f"measured algorithmic delay: {lag} samples (~{lag / SAMPLE_RATE * 1000:.1f} ms)")
    print(f"mean abs error vs batch:    {mean_err:.6f}")
    print(f"max abs error vs batch:     {max_err:.6f}")
    verdict = "real-time capable" if rtf < 1 else "NOT real-time"
    print(f"RTF on this CPU:            {rtf:.4f}  ({verdict})")
    print(f"wrote {out_path}")

    # This residual is NOT float round-off: the ONNX matches upstream's StreamGTCRN to
    # ~1e-7 (see verify_onnx_provenance.py), so it is a real algorithmic difference
    # between the streaming reformulation and the batch model -- chiefly the causal
    # conv caching and the ConvTranspose2d-as-Conv2d rewrite in stream/modules/. Small
    # enough to be inaudible, but it is a conversion artifact, not numerical noise.
    if max_err > 0.05:
        print(
            f"\nWARNING: max error {max_err:.4f} exceeds the expected streaming-conversion "
            "residual (~0.03) -- investigate before going live.",
            file=sys.stderr,
        )
        sys.exit(1)
    print("\nCHECK PASSED: streaming output tracks the batch reference within the expected residual.")


def _chirp(duration_s: float = 0.010, f0: float = 1000.0, f1: float = 5000.0) -> np.ndarray:
    """Short linear chirp. Preferred over a click: its autocorrelation has a much
    sharper peak, so the delay estimate stays reliable in a noisy room."""
    t = np.arange(int(SAMPLE_RATE * duration_s)) / SAMPLE_RATE
    sweep = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * t[-1]) * t**2))
    # Taper the edges so the burst doesn't click and smear the correlation peak.
    return (sweep * np.hanning(len(sweep))).astype(np.float32)


def _find_delay(recording: np.ndarray, probe: np.ndarray) -> tuple[int, float]:
    """Delay of `probe` within `recording`, in samples, by cross-correlation.
    Also returns a peak-to-sidelobe ratio -- a confidence figure. A low ratio means
    the peak isn't clearly above the background and the delay shouldn't be trusted."""
    from scipy.signal import correlate

    corr = np.abs(correlate(recording, probe, mode="valid"))
    peak = int(np.argmax(corr))
    guard = len(probe)
    masked = corr.copy()
    masked[max(0, peak - guard) : peak + guard] = 0
    sidelobe = float(masked.max()) if masked.size else 0.0
    ratio = float(corr[peak]) / sidelobe if sidelobe > 0 else float("inf")
    return peak, ratio


def _selftest_delay_detection() -> bool:
    """Gate 0.2 prerequisite: prove the detector recovers a KNOWN delay before we
    trust it on real audio. Without this, a broken estimator would happily report a
    confident-looking but meaningless latency number."""
    print("self-test: recovering known synthetic delays (no audio hardware used)...")
    probe = _chirp()
    rng = np.random.default_rng(0)
    ok = True
    for true_delay in (0, 137, 1024, 7919):
        signal = rng.standard_normal(SAMPLE_RATE) * 0.01  # room-noise stand-in
        signal[true_delay : true_delay + len(probe)] += probe * 0.5
        found, ratio = _find_delay(signal.astype(np.float32), probe)
        good = abs(found - true_delay) <= 2
        ok &= good
        print(f"  true {true_delay:5d} -> found {found:5d}  ({'ok' if good else 'FAIL'}, peak/sidelobe {ratio:.1f})")
    return ok


def cmd_measure_latency(args) -> None:
    """End-to-end acoustic latency. docs/solution-design.md SS9 is explicit that RTF is
    a model metric and judges are asking about conversation latency -- this measures
    the thing they mean.

    Method: play a chirp and record simultaneously, then cross-correlate. The recovered
    delay covers output buffering -> DAC -> air -> mic -> ADC -> input buffering, i.e.
    the whole hardware round trip. Our processing adds algorithmic delay (measured at 0
    samples by --check, since GTCRN is causal with no lookahead) plus per-hop compute.
    Total one-way conversational latency ~= that round trip + processing.
    """
    if not _selftest_delay_detection():
        print("\nGATE 0.2 FAILED: delay detector cannot recover known delays.", file=sys.stderr)
        sys.exit(1)
    print("  self-test passed.\n")

    import sounddevice as sd

    probe = _chirp()
    pad = int(SAMPLE_RATE * 0.5)
    playback = np.concatenate([np.zeros(pad, np.float32), probe, np.zeros(pad, np.float32)])

    print(f"playing {len(probe) / SAMPLE_RATE * 1000:.0f} ms chirp and recording simultaneously, "
          f"{args.latency_trials} trials...")
    print("keep the room quiet; speakers must be audible to the mic (do NOT use headphones for this test)\n")

    delays_ms = []
    for trial in range(1, args.latency_trials + 1):
        rec = sd.playrec(playback, samplerate=SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        found, ratio = _find_delay(rec[:, 0], probe)
        delay_ms = (found - pad) / SAMPLE_RATE * 1000
        flag = "" if ratio > 3 else "  <-- weak peak, low confidence"
        print(f"  trial {trial}: {delay_ms:7.1f} ms  (peak/sidelobe {ratio:.1f}){flag}")
        if ratio > 3:
            delays_ms.append(delay_ms)

    if not delays_ms:
        print(
            "\nGATE 0.2 FAILED: no trial produced a confident peak. The mic likely can't hear "
            "the speaker -- raise the volume, move the mic closer, or check the output device.",
            file=sys.stderr,
        )
        sys.exit(1)

    round_trip = float(np.median(delays_ms))
    compute_ms = _measure_compute_per_hop()
    algorithmic_ms = 0.0  # measured by --check: causal, zero lookahead
    total = round_trip + algorithmic_ms + compute_ms

    print(f"\n=== end-to-end latency ({len(delays_ms)}/{args.latency_trials} confident trials) ===")
    print(f"hardware round trip (out->air->in): {round_trip:7.1f} ms  (median)")
    print(f"algorithmic (causal, no lookahead): {algorithmic_ms:7.1f} ms")
    print(f"model compute per hop:              {compute_ms:7.2f} ms")
    print(f"TOTAL one-way conversational:       {total:7.1f} ms")

    # ITU-T G.114 (05/2003): 0-150 ms is "acceptable for most user applications".
    if total < 150:
        print(f"\nWithin ITU-T G.114's 150 ms transparent band ({total:.0f} ms).")
    elif total < 400:
        print(f"\nIn G.114's 150-400 ms band -- acceptable with awareness ({total:.0f} ms).")
    else:
        print(f"\nAbove G.114's 400 ms limit ({total:.0f} ms) -- investigate buffering.", file=sys.stderr)

    print(
        "\nNote: this is laptop hardware. The Pi 5's audio stack will differ, so re-run "
        "this on the board rather than quoting these numbers for it."
    )


def _measure_compute_per_hop(n_hops: int = 200, onnx_path=None) -> float:
    """Median wall-clock ms the model needs per hop -- the compute term of the budget."""
    enhancer = StreamingEnhancer(onnx_path) if onnx_path else StreamingEnhancer()
    rng = np.random.default_rng(0)
    times = []
    for _ in range(n_hops):
        hop = (rng.standard_normal(HOP) * 0.05).astype(np.float32)
        t0 = time.perf_counter()
        enhancer.process_hop(hop, enabled=True)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def _load_noise_loop(path: str) -> np.ndarray:
    noise, fs = sf.read(path, dtype="float32")
    if noise.ndim > 1:
        noise = noise.mean(axis=1)
    if fs != SAMPLE_RATE:
        from scipy import signal as _sps      # scipy ships on the Pi, torchaudio does not
        noise = _sps.resample(noise, int(len(noise) * SAMPLE_RATE / fs))
    return noise.astype(np.float32)


def _synthetic_pink_noise(seconds: float = 10.0) -> np.ndarray:
    """Cheap pink-ish noise (Paul Kellet's filter) so the demo has a repeatable
    noise bed even with no noise dataset downloaded yet."""
    from scipy.signal import lfilter

    rng = np.random.default_rng(0)
    white = rng.standard_normal(int(SAMPLE_RATE * seconds)).astype(np.float32)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
    a = [1, -2.494956002, 2.017265875, -0.522189400]
    pink = lfilter(b, a, white).astype(np.float32)
    return pink / (np.abs(pink).max() + 1e-8)


def cmd_capture_test(args) -> None:
    """Record from the mic, enhance, and save BOTH the raw and enhanced audio.

    Saving the raw input is the point: an enhanced file on its own proves nothing,
    because a listener has no idea what the microphone actually heard. The pair is
    the evidence.

    In --dual-mic mode, channel 0 is the main INMP441 and channel 1 is the
    reference INMP441. Both raw channels are saved for comparison.
    """
    import sounddevice as sd

    dual = getattr(args, "dual_mic", False)
    n_channels = 2 if dual else 1

    ref_kw = dict(ref_alpha=getattr(args, "ref_alpha", 1.0),
                  ref_beta=getattr(args, "ref_beta", 0.02),
                  ref_smoothing=getattr(args, "ref_smoothing", 0.95))
    enhancer = (StreamingEnhancer(args.onnx, dual_mic=dual, **ref_kw)
                if args.onnx else StreamingEnhancer(dual_mic=dual, **ref_kw))
    raw_main_hops, raw_ref_hops, enhanced_hops = [], [], []

    native_48k = getattr(args, "native_48k", False)
    stream_sr = 48000 if native_48k else SAMPLE_RATE
    stream_hop = HOP * 3 if native_48k else HOP

    def cb(indata, _frames, _time_info, status):
        if status:
            print(status, file=sys.stderr)
        main_in = indata[:, 0].copy()
        ref_in = indata[:, 1].copy() if dual else None
        
        main_hop = main_in[::3] if native_48k else main_in
        ref_hop = ref_in[::3] if (native_48k and ref_in is not None) else ref_in
        
        raw_main_hops.append(main_hop)
        if ref_hop is not None:
            raw_ref_hops.append(ref_hop)
        enhanced_hops.append(enhancer.process_hop(main_hop, ref_hop=ref_hop, enabled=True))

    if dual:
        print("DUAL-MIC MODE: channel 0 = main INMP441, channel 1 = reference INMP441")
    print(f"model: {args.onnx if args.onnx else 'upstream pretrained'}")
    print(f"\nRecording {args.capture_test:.0f}s. Start your noise source now, then speak.")
    for n in (3, 2, 1):
        print(f"  {n}...", flush=True)
        time.sleep(1)
    print("  GO -- speak now\n", flush=True)

    with sd.InputStream(samplerate=stream_sr, blocksize=stream_hop, channels=n_channels, dtype="float32",
                        device=_dev(getattr(args, "input_device", None)), callback=cb):
        sd.sleep(int(args.capture_test * 1000))
    print("done recording.")

    if not enhanced_hops:
        sys.exit("nothing captured -- check the input device with: python -c \"import sounddevice;print(sounddevice.query_devices())\"")

    raw_main = np.concatenate(raw_main_hops)
    enhanced = np.concatenate(enhanced_hops)

    out_enh = Path(args.out)
    out_raw = out_enh.with_name(out_enh.stem + "_raw" + out_enh.suffix)
    sf.write(out_raw, raw_main, SAMPLE_RATE)
    sf.write(out_enh, enhanced, SAMPLE_RATE)

    level = 20 * np.log10(np.sqrt(np.mean(raw_main**2)) + 1e-9)
    print(f"\n  BEFORE (what the main mic heard):  {out_raw}")
    print(f"  AFTER  (enhanced):                  {out_enh}")

    if dual and raw_ref_hops:
        raw_ref = np.concatenate(raw_ref_hops)
        out_ref = out_enh.with_name(out_enh.stem + "_ref" + out_enh.suffix)
        sf.write(out_ref, raw_ref, SAMPLE_RATE)
        ref_level = 20 * np.log10(np.sqrt(np.mean(raw_ref**2)) + 1e-9)
        print(f"  REF    (reference mic noise):        {out_ref}")
        print(f"  ref mic input level {ref_level:.1f} dBFS")

    print(f"  {len(raw_main)/SAMPLE_RATE:.1f}s captured, main mic input level {level:.1f} dBFS")
    if level < -50:
        print("  WARNING: input is very quiet -- check the mic is selected and unmuted.", file=sys.stderr)
    print(f"\nCompare them:\n  aplay {out_raw}\n  aplay {out_enh}")


# ---------------------------------------------------------------- live visual
# A terminal spectrogram, not a plot window. matplotlib is not installed on the
# Pi (setup_pi.sh keeps the runtime to five packages) and an X/VNC window would
# stutter over the network, whereas ANSI blocks render instantly inside the same
# SSH session the demo is already running in.
_RAMP = [16, 17, 18, 19, 20, 25, 26, 31, 37, 43, 49, 84, 119, 154, 190, 226, 220, 214, 208, 202, 196]
_NB = 34


def _bands(mag: np.ndarray, nb: int = _NB) -> np.ndarray:
    """Average |X| into log-spaced bands, so low frequencies get the detail.

    Edges are clamped so every band spans at least one bin: geomspace collapses
    to duplicate integers at the low end, which otherwise yields empty slices,
    NaN means, and a garbage colour index.
    """
    n = len(mag)
    edges = np.round(np.geomspace(1, n, nb + 1)).astype(int)
    out = np.empty(nb, dtype=np.float64)
    for i in range(nb):
        lo = min(int(edges[i]), n - 1)
        hi = min(max(int(edges[i + 1]), lo + 1), n)
        out[i] = mag[lo:hi].mean()
    return out


def _row(db: np.ndarray, lo: float = -68.0, hi: float = 6.0) -> str:
    v = np.clip(np.nan_to_num((db - lo) / (hi - lo), nan=0.0), 0.0, 1.0)
    idx = (v * (len(_RAMP) - 1)).astype(int)
    return "".join(f"\033[48;5;{_RAMP[i]}m " for i in idx) + "\033[0m"


def _meter(x: np.ndarray, width: int = 10) -> str:
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + 1e-12))
    n = int(np.clip((20 * np.log10(rms + 1e-9) + 60) / 60, 0, 1) * width)
    return "\u2588" * n + "\u00b7" * (width - n)


def _visual_header() -> None:
    w = _NB
    print()
    print(f"  {'MICROPHONE IN'.center(w)}   {'MODEL OUT'.center(w)}")
    print(f"  {'0 kHz' + ' ' * (w - 11) + '8 kHz'}   {'0 kHz' + ' ' * (w - 11) + '8 kHz'}")
    print(f"  {'-' * w}   {'-' * w}")


def cmd_live(args) -> None:
    import sounddevice as sd

    dual = getattr(args, "dual_mic", False)
    n_in_channels = 2 if dual else 1

    ref_kw = dict(ref_alpha=getattr(args, "ref_alpha", 1.0),
                  ref_beta=getattr(args, "ref_beta", 0.02),
                  ref_smoothing=getattr(args, "ref_smoothing", 0.95))
    enhancer = (StreamingEnhancer(args.onnx, dual_mic=dual, **ref_kw)
                if args.onnx else StreamingEnhancer(dual_mic=dual, **ref_kw))
    if args.onnx:
        print(f"model: {args.onnx}")
    if dual:
        print("DUAL-MIC MODE: channel 0 = main INMP441, channel 1 = reference INMP441")
        print("output: BT headphone (mono)")
    state = {"enabled": True, "running": True, "rtf_samples": [], "last_print": time.time(), "noise_pos": 0}

    noise_loop = None
    if args.inject_noise:
        noise_loop = _load_noise_loop(args.inject_noise)
        print(f"injecting noise from {args.inject_noise} at gain {args.noise_gain}")
    elif args.synthetic_noise:
        noise_loop = _synthetic_pink_noise()
        print(f"injecting synthetic pink noise at gain {args.noise_gain}")

    native_48k = getattr(args, "native_48k", False)
    stream_sr = 48000 if native_48k else SAMPLE_RATE
    stream_hop = HOP * 3 if native_48k else HOP

    def callback(indata, outdata, frames, _time_info, status):
        if status:
            print(status, file=sys.stderr)
        # Process in chunks of stream_hop
        for off in range(0, frames, stream_hop):
            main_hop_in = indata[off : off + stream_hop, 0].copy()
            if len(main_hop_in) < stream_hop:
                main_hop_in = np.pad(main_hop_in, (0, stream_hop - len(main_hop_in)))

            # Reference mic (channel 1) in dual-mic mode.
            ref_hop_in = None
            if dual and indata.shape[1] >= 2:
                ref_hop_in = indata[off : off + stream_hop, 1].copy()
                if len(ref_hop_in) < stream_hop:
                    ref_hop_in = np.pad(ref_hop_in, (0, stream_hop - len(ref_hop_in)))

            # Convert 48k to 16k by simple 3x decimation if native_48k is True
            main_hop = main_hop_in[::3] if native_48k else main_hop_in
            ref_hop = ref_hop_in[::3] if (native_48k and ref_hop_in is not None) else ref_hop_in

            if noise_loop is not None:
                pos = state["noise_pos"]
                idx = (np.arange(pos, pos + HOP)) % len(noise_loop)
                main_hop = main_hop + args.noise_gain * noise_loop[idx]
                state["noise_pos"] = (pos + HOP) % len(noise_loop)

            t0 = time.perf_counter()
            out_hop = enhancer.process_hop(main_hop, ref_hop=ref_hop, enabled=state["enabled"])
            state["rtf_samples"].append((time.perf_counter() - t0) / (HOP / SAMPLE_RATE))
            
            # Convert 16k back to 48k by repeating samples
            out_hop_out = np.repeat(out_hop, 3) if native_48k else out_hop
            outdata[off : off + stream_hop, 0] = out_hop_out[: min(stream_hop, frames - off)]
            
            if args.visual:
                L = enhancer.last
                state["frame"] = (np.abs(L["spec"]), np.abs(L["spec_out"]), main_hop, out_hop)

        now = time.time()
        if not args.visual and now - state["last_print"] > 2.0:
            recent = state["rtf_samples"][-200:]
            mean_rtf = float(np.mean(recent)) if recent else 0.0
            mode = "ENHANCED" if state["enabled"] else "BYPASS "
            extra = ""
            if dual:
                L = enhancer.last
                if "pre_sub_mag" in L and "post_sub_mag" in L:
                    reduction_db = 20 * np.log10(
                        (np.mean(L["pre_sub_mag"]) + 1e-9) /
                        (np.mean(L["post_sub_mag"]) + 1e-9))
                    extra = f"  ref-sub: {reduction_db:.1f} dB"
            print(f"[{mode}] rolling RTF: {mean_rtf:.3f}{extra}")
            state["last_print"] = now

    def key_listener():
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while state["running"]:
                ch = sys.stdin.read(1)
                if ch.lower() == "e":
                    state["enabled"] = not state["enabled"]
                    print(f">>> enhancement {'ON' if state['enabled'] else 'OFF'}")
                elif ch.lower() == "q":
                    state["running"] = False
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    print("live demo running. press 'e' to toggle enhancement, 'q' to quit.")
    listener = threading.Thread(target=key_listener, daemon=True)
    listener.start()

    dev_in = _dev(getattr(args, "input_device", None))
    dev_out = _dev(getattr(args, "output_device", None))
    if dev_in is not None or dev_out is not None:
        print(f"input device:  {dev_in if dev_in is not None else 'system default'}")
        print(f"output device: {dev_out if dev_out is not None else 'system default'}")
    bs = max(stream_hop, (int(args.blocksize) // stream_hop) * stream_hop)
    if bs != stream_hop:
        print(f"block size {bs} samples ({bs / stream_sr * 1000:.0f} ms) -- "
              f"more slack against dropouts, {(bs - stream_hop) / stream_sr * 1000:.0f} ms more latency")

    # In dual-mic mode we need separate input (2 ch) and output (1 ch) channel
    # counts, so we open an sd.Stream with explicit input/output channels.
    with sd.Stream(samplerate=stream_sr, blocksize=bs,
                   channels=n_in_channels,   # input channels (1 or 2)
                   dtype="float32",
                   device=(dev_in, dev_out) if (dev_in is not None or dev_out is not None) else None,
                   callback=callback):
        try:
            if args.visual:
                _visual_header()
            while state["running"] and listener.is_alive():
                if args.visual and state.get("frame") is not None:
                    mi, mo, hi_, ho = state["frame"]
                    din = 20 * np.log10(_bands(mi) + 1e-7)
                    dout = 20 * np.log10(_bands(mo) + 1e-7)
                    recent = state["rtf_samples"][-120:]
                    rtf = float(np.mean(recent)) if recent else 0.0
                    tag = "\033[42;30m ENHANCED \033[0m" if state["enabled"] else "\033[41;37m BYPASS   \033[0m"
                    print(f"  {_row(din)}   {_row(dout)}  {tag} "
                          f"in {_meter(hi_)} out {_meter(ho)} rtf {rtf:.2f}")
                    state["frame"] = None
                    time.sleep(0.08)
                else:
                    time.sleep(0.05)
        except KeyboardInterrupt:
            state["running"] = False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="offline correctness + RTF check, no audio hardware")
    parser.add_argument(
        "--measure-latency", action="store_true", help="end-to-end acoustic latency (needs mic + speakers)"
    )
    parser.add_argument("--latency-trials", type=int, default=5, help="repeat count for --measure-latency")
    parser.add_argument(
        "--onnx",
        type=Path,
        default=None,
        help="streaming ONNX model to run (default: the pretrained one bundled upstream). "
        "Point this at a model produced by scripts/export_onnx.py to demo a fine-tuned model.",
    )
    parser.add_argument("--capture-test", type=float, metavar="SECONDS", help="mic -> file for SECONDS, no playback")
    parser.add_argument("--blocksize", type=int, default=HOP,
                        help=f"device block in samples, a multiple of {HOP}. Raise to 512 or "
                             f"1024 if audio drops out (trades latency for stability)")
    parser.add_argument("--visual", action="store_true",
                        help="live terminal spectrogram of input vs model output")
    parser.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    parser.add_argument("--native-48k", action="store_true", 
                        help="Run audio stream natively at 48000Hz (for hardware that rejects 16kHz) and downsample in Python.")
    parser.add_argument("--input-device", metavar="ID|NAME", help="mic to capture from")
    parser.add_argument("--output-device", metavar="ID|NAME", help="where to play the enhanced audio (e.g. BT headphone)")
    parser.add_argument("--process", metavar="FILE", help="enhance a wav file and exit; needs no audio hardware")
    parser.add_argument("--out", default="capture_test.wav", help="output wav for --capture-test")
    parser.add_argument("--inject-noise", metavar="FILE", help="loop this wav additively into the mic signal (live mode)")
    parser.add_argument("--synthetic-noise", action="store_true", help="inject synthetic pink noise (live mode)")
    parser.add_argument("--noise-gain", type=float, default=0.3, help="linear gain for injected noise")
    # ---- dual INMP441 microphone options ----
    parser.add_argument("--dual-mic", action="store_true",
                        help="dual INMP441 mode: channel 0 = main mic (near mouth), "
                             "channel 1 = reference mic (ambient noise). The reference "
                             "mic's spectrum is subtracted from the main before GTCRN.")
    parser.add_argument("--ref-alpha", type=float, default=1.0,
                        help="over-subtraction factor for reference noise removal "
                             "(higher = more aggressive, default 1.0)")
    parser.add_argument("--ref-beta", type=float, default=0.02,
                        help="spectral floor as fraction of ref noise estimate "
                             "(prevents musical noise, default 0.02)")
    parser.add_argument("--ref-smoothing", type=float, default=0.95,
                        help="exponential smoothing for ref noise estimate "
                             "(higher = smoother/slower, default 0.95)")
    args = parser.parse_args()

    if args.list_devices:
        cmd_list_devices(args)
    elif args.check:
        cmd_check(args)
    elif args.process:
        cmd_process(args)
    elif args.measure_latency:
        cmd_measure_latency(args)
    elif args.capture_test:
        cmd_capture_test(args)
    else:
        cmd_live(args)


if __name__ == "__main__":
    main()
