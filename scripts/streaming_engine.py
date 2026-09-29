"""
Frame-by-frame streaming GTCRN inference with dual INMP441 microphone support.

Uses the pre-exported ONNX model at
third_party/gtcrn/stream/onnx_models/gtcrn_simple.onnx (upstream, MIT licence,
exported by the GTCRN authors from checkpoints/model_trained_on_dns3.tar). This
module only calls that model and handles the STFT/overlap-add framing around it
-- third_party/gtcrn is not modified.

Dual-mic mode (military config):
  - Channel 0: Main INMP441 mic (close to the speaker's mouth)
  - Channel 1: Reference INMP441 mic (captures ambient battlefield noise)
  The reference channel's spectrum is used for spectral-domain noise estimation.
  The estimated noise is subtracted from the main mic's spectrum before GTCRN
  processes it, giving the model a cleaner input and improving enhancement in
  extreme noise environments (gunshots, shelling, vehicle noise).
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = REPO_ROOT / "third_party" / "gtcrn" / "stream" / "onnx_models" / "gtcrn_simple.onnx"

N_FFT = 512
HOP = 256

# Periodic sqrt-Hann window, matching torch.hann_window(512, periodic=True).pow(0.5)
# (the convention used throughout third_party/gtcrn, including training).
_n = np.arange(N_FFT)
WINDOW = np.sqrt(0.5 - 0.5 * np.cos(2 * np.pi * _n / N_FFT)).astype(np.float32)


class StreamingEnhancer:
    """Carries GTCRN's recurrent/conv state across process_hop() calls.

    Supports two modes:
      1. Single-mic: process_hop(hop) -- backward compatible, original behaviour.
      2. Dual-mic:   process_hop(main_hop, ref_hop) -- reference-aided noise
         subtraction before GTCRN enhancement.
    """

    def __init__(self, onnx_path: Path = ONNX_PATH, dual_mic: bool = False,
                 ref_alpha: float = 1.0, ref_beta: float = 0.02,
                 ref_smoothing: float = 0.95):
        """
        Args:
            onnx_path:      path to the streaming ONNX model.
            dual_mic:       if True, process_hop() expects two arrays (main, ref).
            ref_alpha:      over-subtraction factor for reference-based noise removal.
                            Higher values remove more noise but risk speech distortion.
            ref_beta:       spectral floor as a fraction of the noise estimate;
                            prevents deep spectral nulls that cause musical noise.
            ref_smoothing:  exponential smoothing factor for the running noise
                            estimate derived from the reference mic.
        """
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.dual_mic = dual_mic
        self.ref_alpha = ref_alpha
        self.ref_beta = ref_beta
        self.ref_smoothing = ref_smoothing
        self.reset()

    def reset(self) -> None:
        self.conv_cache = np.zeros((2, 1, 16, 16, 33), dtype=np.float32)
        self.tra_cache = np.zeros((2, 3, 1, 1, 16), dtype=np.float32)
        self.inter_cache = np.zeros((2, 1, 33, 16), dtype=np.float32)
        self.analysis_buf = np.zeros(N_FFT, dtype=np.float32)
        self.synthesis_buf = np.zeros(N_FFT, dtype=np.float32)
        # Reference mic analysis buffer and running noise magnitude estimate.
        self.ref_analysis_buf = np.zeros(N_FFT, dtype=np.float32)
        self.ref_noise_mag: np.ndarray | None = None  # initialised on first frame
        # Per-hop intermediates, kept for introspection (scripts/make_pipeline_animation.py
        # renders them). These are references to arrays this method already builds, so
        # populating them costs nothing and changes no behaviour.
        self.last: dict = {}

    def _ref_subtract(self, main_spec: np.ndarray, ref_hop: np.ndarray) -> np.ndarray:
        """Spectral subtraction using the reference mic's magnitude spectrum.

        The reference INMP441 captures mostly ambient noise (it is placed away
        from the speaker's mouth). Its magnitude spectrum provides a live,
        per-frame noise estimate that is far more accurate than the blind
        leading-silence estimate used in classical spectral subtraction.

        Returns a complex spectrum with the noise contribution attenuated.
        """
        # STFT of the reference hop (same window/FFT parameters).
        self.ref_analysis_buf = np.concatenate([self.ref_analysis_buf[HOP:], ref_hop])
        ref_windowed = self.ref_analysis_buf * WINDOW
        ref_spec = np.fft.rfft(ref_windowed, n=N_FFT)
        ref_mag = np.abs(ref_spec)

        # Exponentially smooth the reference magnitude to avoid frame-to-frame
        # jitter. The first frame bootstraps the estimate.
        if self.ref_noise_mag is None:
            self.ref_noise_mag = ref_mag.copy()
        else:
            self.ref_noise_mag = (self.ref_smoothing * self.ref_noise_mag
                                  + (1.0 - self.ref_smoothing) * ref_mag)

        # Over-subtraction with a spectral floor (Boll 1979 + Martin 2001).
        main_mag = np.abs(main_spec)
        main_phase = np.angle(main_spec)
        subtracted = main_mag - self.ref_alpha * self.ref_noise_mag
        floor = self.ref_beta * self.ref_noise_mag
        clean_mag = np.maximum(subtracted, floor)

        # Store for introspection.
        self.last["ref_mag"] = ref_mag
        self.last["ref_noise_mag"] = self.ref_noise_mag.copy()
        self.last["pre_sub_mag"] = main_mag
        self.last["post_sub_mag"] = clean_mag

        return clean_mag * np.exp(1j * main_phase)

    def process_hop(self, hop: np.ndarray, ref_hop: np.ndarray | None = None,
                    enabled: bool = True) -> np.ndarray:
        """Process one hop (256 samples) from the main mic, optionally aided by
        the reference mic.

        Args:
            hop:      (HOP,) float32 from the main INMP441 mic.
            ref_hop:  (HOP,) float32 from the reference INMP441 mic, or None
                      for single-mic mode.
            enabled:  if False the model still runs (keeps state warm) but the
                      output is the unenhanced main signal.

        Returns:
            (HOP,) float32 enhanced audio.

        The model always runs (state stays warm even while bypassed) so toggling
        `enabled` mid-stream doesn't cause a stale-hidden-state glitch when
        re-enabled -- only the reconstructed spectrum choice depends on the flag.
        """
        if hop.shape != (HOP,):
            raise ValueError(f"expected hop shape ({HOP},), got {hop.shape}")
        if ref_hop is not None and ref_hop.shape != (HOP,):
            raise ValueError(f"expected ref_hop shape ({HOP},), got {ref_hop.shape}")

        # --- analysis: main mic ---
        self.analysis_buf = np.concatenate([self.analysis_buf[HOP:], hop])
        windowed = self.analysis_buf * WINDOW
        spec = np.fft.rfft(windowed, n=N_FFT)  # (257,) complex64

        # --- reference-aided spectral subtraction (dual-mic mode) ---
        if ref_hop is not None and self.dual_mic:
            spec_for_model = self._ref_subtract(spec, ref_hop)
        else:
            spec_for_model = spec

        mix = np.stack([spec_for_model.real, spec_for_model.imag],
                       axis=-1).astype(np.float32)[None, :, None, :]  # (1,257,1,2)
        enh, self.conv_cache, self.tra_cache, self.inter_cache = self.session.run(
            None,
            {
                "mix": mix,
                "conv_cache": self.conv_cache,
                "tra_cache": self.tra_cache,
                "inter_cache": self.inter_cache,
            },
        )
        enh = enh[0, :, 0, :]  # (257, 2)
        spec_out = (enh[:, 0] + 1j * enh[:, 1]) if enabled else spec

        frame = np.fft.irfft(spec_out, n=N_FFT).astype(np.float32) * WINDOW
        self.synthesis_buf = self.synthesis_buf + frame
        out_hop = self.synthesis_buf[:HOP].copy()
        self.synthesis_buf = np.concatenate([self.synthesis_buf[HOP:], np.zeros(HOP, dtype=np.float32)])
        self.last.update({"windowed": windowed, "spec": spec,
                          "spec_for_model": spec_for_model,
                          "spec_out": spec_out, "frame": frame,
                          "out_hop": out_hop})
        return out_hop
