# SIH-DRDO: Real-Time AI Speech Enhancement for Defence Communications

### Complete Technical Documentation

---

## 1. Project Overview

Military personnel operating in tactical battlefield environments confront extreme acoustic interference, including high-caliber automatic gunfire, low-frequency helicopter rotor wash, armored vehicle diesel powertrain noise, and sudden supersonic artillery detonations. Under these mission-critical conditions, conventional tactical radio links suffer catastrophic speech intelligibility degradation. Standard hardware or classical digital noise-cancellation systems rely heavily on acoustic stationarity assumptions; consequently, they fail abruptly when subjected to high-energy impulsive transients and wideband shock waves, introducing severe speech distortion, phase jitter, and musical noise artifacts.

To solve this operational failure, we have engineered an ultra-lightweight, edge-native speech enhancement system anchored by a 23.67K-parameter Grouped Temporal Convolutional Recurrent Network ([`GTCRN`](file:///Users/nirajrajendranaphade/Programming/sih-drdo/third_party/gtcrn/gtcrn.py#L182-L245)) executing entirely on CPU hardware (Raspberry Pi 4 / Raspberry Pi 5). The acoustic frontend utilizes a synchronized dual-microphone MEMS array ([INMP441](https://invensense.tdk.com/products/digital/inmp441/)) interfaced directly over the digital I2S serial bus. A primary mouth-facing microphone captures near-field speech combined with ambient interference, while an outward-facing reference microphone measures the ambient battlefield soundscape. A real-time, low-complexity spectral subtraction pre-processing stage dynamically attenuates wideband steady-state energy before the complex spectral tensor reaches the neural network, allowing the deep learning model to focus its inference capacity on non-linear residual transients, phase alignment, and speech formant restoration.

The resulting hybrid DSP-AI system achieves unprecedented computational efficiency and acoustic quality on resource-constrained embedded edge hardware. On a Raspberry Pi 5, the inference pipeline achieves a Real-Time Factor (RTF) of **0.099** (utilizing only 1.59 ms of single-threaded compute per 16.0 ms audio hop, providing a 10× real-time processing headroom) with an acoustic round-trip latency of **83.6 ms**, comfortably below the 150 ms threshold prescribed by [ITU-T Recommendation G.114](https://www.itu.int/rec/T-REC-G.114). Rigorous objective evaluations on real-world military audio under hostile conditions (+15 dB down to −5 dB SNR) demonstrate a Perceptual Evaluation of Speech Quality (PESQ) of **2.49 ± 0.04**, Scale-Invariant Signal-to-Noise Ratio (SI-SNR) of **20.2 dB**, and Short-Time Objective Intelligibility (STOI) of **0.920** on challenging impulsive transients—reversing the historic vulnerability of deep models to impulsive battlefield noise.

---

## 2. System Architecture

The end-to-end signal processing architecture operates strictly causally without future frame lookahead, preserving a zero-algorithmic-delay profile across every stage of the pipeline.

```mermaid
flowchart TD
    subgraph Hardware_Frontend["Acoustic Hardware Frontend"]
        M1["Main Mic (INMP441 MEMS)<br/>Near-mouth: Speech + Battlefield Noise"]
        M2["Reference Mic (INMP441 MEMS)<br/>Outward-facing: Ambient Battlefield Noise"]
        I2S["Digital I2S Audio Bus<br/>Left/Right Time-Multiplexed Stereo"]
        M1 -->|Channel 0 (Left)| I2S
        M2 -->|Channel 1 (Right)| I2S
    end

    subgraph Host_Processing["Raspberry Pi 4 / 5 Edge Compute (ARM Cortex CPU)"]
        ALSA["ALSA / PortAudio Audio Driver<br/>Native 48 kHz / 16 kHz Ingestion"]
        I2S -->|Raw PCM Stream| ALSA

        subgraph DSP_Pre_Stage["DSP Pre-Processing Stage"]
            RB["Frame Ring Buffer<br/>32 ms Window (512 samples)<br/>16 ms Hop (256 samples)"]
            STFT["STFT Spectral Analysis<br/>sqrt-Hann Window & 512-point FFT"]
            SPEC_SUB["Dual-Mic Spectral Subtraction<br/>Dynamic Over-subtraction & Floor<br/>Over-subtraction factor α, Floor β"]
            
            ALSA -->|Main + Ref PCM| RB
            RB --> STFT
            STFT -->|Main & Ref Spectra| SPEC_SUB
        end

        subgraph Neural_Engine["GTCRN Neural Inference Engine (ONNX Runtime)"]
            ERB["ERB Filterbank Compression<br/>257 Bins → 129 Psychoacoustic Bands"]
            ENC["Grouped Conv Encoder<br/>SFE + GTConvBlocks + Channel Shuffle"]
            RNN["Dual-Path Grouped RNN (DPGRNN)<br/>Intra-RNN (Freq) & Inter-RNN (Causal Time)"]
            DEC["Transposed Conv Decoder<br/>Skip Connections from Encoder"]
            MASK["Complex Ratio Mask (CRM)<br/>Magnitude Masking + Phase Reconstruction"]

            SPEC_SUB -->|Pre-Cleaned Complex Spectrum| ERB
            ERB --> ENC
            ENC --> RNN
            RNN --> DEC
            DEC --> MASK
            RNN -.->|Carried Hidden State<br/>(Between Hops)| RNN
        end

        subgraph Synthesis_Stage["Synthesis & Output Engine"]
            ISTFT["iSTFT Synthesis & Overlap-Add<br/>Reconstruct 256-sample Audio Hop"]
            RING_OUT["Output Audio Buffer<br/>Glitch-Free Streaming Queue"]
            
            MASK --> ISTFT
            ISTFT --> RING_OUT
        end
    end

    subgraph Audio_Sink["Tactical Output"]
        BT["Tactical Bluetooth Headset /<br/>VHF/UHF Radio Transmitter"]
        RING_OUT -->|Clean 16 kHz Mono Audio| BT
    end

    style Hardware_Frontend fill:#1f2937,stroke:#3b82f6,stroke-width:2px,color:#fff
    style Host_Processing fill:#111827,stroke:#10b981,stroke-width:2px,color:#fff
    style DSP_Pre_Stage fill:#1e293b,stroke:#0ea5e9,stroke-width:1px,color:#fff
    style Neural_Engine fill:#1e293b,stroke:#8b5cf6,stroke-width:1px,color:#fff
    style Synthesis_Stage fill:#1e293b,stroke:#0ea5e9,stroke-width:1px,color:#fff
    style Audio_Sink fill:#1f2937,stroke:#f59e0b,stroke-width:2px,color:#fff
```

### Signal Processing Flow Breakdown

1. **Acoustic Transduction & I2S Ingestion:**
   The hardware frontend pairs two omnidirectional [`INMP441`](https://invensense.tdk.com/products/digital/inmp441/) MEMS sensors sharing a single serial clock (SCK) and word select (WS) line. By tying the `L/R` channel pin of the primary mic to `GND` and the reference mic to `VDD`, stereo PCM data is multiplexed into the Raspberry Pi I2S bus with cycle-accurate synchronization.
2. **Analysis Framing & STFT Transformation:**
   Input samples pass into a rolling ring buffer. Every hop duration ($H = 256$ samples, $16.0\text{ ms}$ at $f_s = 16\text{ kHz}$), a frame of $N = 512$ samples ($32.0\text{ ms}$) is windowed with a periodic square-root Hann window $w[n]$:
   $$w[n] = \sqrt{0.5 - 0.5 \cos\left(\frac{2\pi n}{N}\right)}$$
   The discrete Fourier transform yields $257$ single-sided complex frequency bins:
   $$X_{\text{main}}[k, t] = \text{DFT}\{x_{\text{main}}[n] \cdot w[n]\}, \quad X_{\text{ref}}[k, t] = \text{DFT}\{x_{\text{ref}}[n] \cdot w[n]\}$$
3. **Reference-Aided Spectral Subtraction Pre-Stage:**
   The reference microphone spectrum provides an ambient acoustic estimate $P_{\text{ref}}[k, t]$, smoothed via an exponential moving average ($\lambda = 0.95$):
   $$P_{\text{ref}}[k, t] = \lambda P_{\text{ref}}[k, t-1] + (1 - \lambda) |X_{\text{ref}}[k, t]|$$
   Over-subtraction with a parameterizable spectral floor factor $\beta = 0.02$ and over-subtraction multiplier $\alpha = 1.0$ suppresses stationary background energy while protecting against musical noise:
   $$|X_{\text{sub}}[k, t]| = \max\Big(|X_{\text{main}}[k, t]| - \alpha P_{\text{ref}}[k, t], \;\; \beta P_{\text{ref}}[k, t]\Big)$$
   $$X_{\text{pre}}[k, t] = |X_{\text{sub}}[k, t]| \cdot \exp\big(j \angle X_{\text{main}}[k, t]\big)$$
4. **Grouped Temporal Convolutional Recurrent Inference:**
   The pre-cleaned complex spectrum is packed into a 3-channel feature representation $[\text{Mag}, \text{Real}, \text{Imag}]$ and mapped through an Equivalent Rectangular Bandwidth (ERB) filterbank ($257 \to 129$ subbands). The convolutional encoder, causal dual-path grouped recurrent units (DPGRNN), and transposed convolutional decoder estimate a bounded Complex Ratio Mask (CRM) $M[k, t] \in \mathbb{C}$.
5. **Mask Application & iSTFT Synthesis:**
   The complex spectrum is filtered by complex multiplication:
   $$\hat{S}[k, t] = X_{\text{pre}}[k, t] \otimes M[k, t]$$
   The inverse DFT converts the enhanced spectrum back to time-domain frames, followed by synthesis windowing and 50% overlap-add reconstruction to stream glitch-free PCM audio to the output hardware sink.

---

## 3. Key Specifications

| Parameter / Metric | Value | Technical Context & Measurement Conditions |
|---|---|---|
| **Learnable Model Parameters** | **23.67 K** | Trainable weights in convolutional blocks, DPGRNN, and TRA gates |
| **Total Model Parameters** | **48.20 K** | Includes fixed non-trainable ERB analysis/synthesis filterbanks |
| **Computational Complexity** | **33.0 MMACs/s** | Million Multiply-Accumulate operations per second (single hop: 0.528 MMACs) |
| **Audio Sample Rate ($f_s$)** | **16,000 Hz** | Standard wideband military communications telephony standard |
| **Analysis Window Length ($N_{\text{FFT}}$)** | **512 samples** | 32.0 ms temporal analysis window |
| **Hop Length ($H$)** | **256 samples** | 16.0 ms frame rate (50% temporal overlap) |
| **Window Type** | **Periodic sqrt-Hann** | Matches training pipeline; achieves perfect reconstruction under 50% overlap-add |
| **Frequency Dimension** | **257 bins $\to$ 129 ERB bands** | 65 linear bins ($0-2\text{ kHz}$) + 64 psychoacoustic bands ($2-8\text{ kHz}$) |
| **Algorithmic Latency** | **0.0 ms** | Strictly causal convolutions and unidirectional recurrent units (0 future lookahead) |
| **Acoustic Round-Trip Latency** | **83.6 ms** | Median measured delay via cross-correlation chirp (within ITU-T G.114 < 150 ms band) |
| **Hardware Compute Time (Pi 5)** | **1.59 ms / hop** | Single-threaded ONNX Runtime execution on ARM Cortex-A76 @ 2.4 GHz |
| **Real-Time Factor (RTF on Pi 5)** | **0.099** | Ratio of compute time to physical audio duration ($1.59\text{ ms} / 16.0\text{ ms}$); 10.1× headroom |
| **PESQ (Wideband, P.862.2)** | **2.49 ± 0.04** | Measured on real military gunfire/shelling at +15 dB SNR (Baseline mixture: 1.78) |
| **Speech Intelligibility (STOI)** | **0.920** | Measured on real impulsive defence audio (DRDO requirement: > 0.85) |
| **Signal-to-Noise Ratio (SI-SNR)** | **20.2 dB** | Scale-Invariant SNR on impulsive battlefield audio (DRDO requirement: > 15 dB) |

---

## 4. Documentation Index

The technical documentation suite is organized modularly to cover every architectural layer, mathematical formulation, hardware configuration, and validation protocol:

| Document | Primary Focus & Core Contents | Target Audience |
|---|---|---|
| [**01 - System Overview**](01_system_overview.md) | High-level system architecture, problem formulation, tactical battlefield noise taxonomy, military operational requirements, end-to-end signal flow, and high-level block descriptions. | Systems Architects, Project Evaluators |
| [**02 - GTCRN Model Architecture**](02_model_architecture.md) | Deep mathematical dive into the Grouped Temporal Convolutional Recurrent Network: ERB filterbank design, Subband Feature Extraction (SFE), GTConvBlock structure, Channel Shuffling, Intra/Inter DPGRNN recurrent cells, and Complex Ratio Mask (CRM) estimation. | Deep Learning Researchers, DSP Engineers |
| [**03 - Streaming Inference Engine**](03_streaming_engine.md) | Low-level execution details of [`StreamingEnhancer`](file:///Users/nirajrajendranaphade/Programming/sih-drdo/scripts/streaming_engine.py#L35-L176): ring buffers, non-centered causal STFT, ONNX Runtime session configuration, recurrent hidden state caches (`conv_cache`, `tra_cache`, `inter_cache`), and zero-latency overlap-add reconstruction. | Embedded Systems Developers, ML Engineers |
| [**04 - Dual Microphone System**](04_dual_mic_system.md) | Acoustic theory of dual-sensor noise cancellation, spatial mic positioning, digital I2S stereo bus multiplexing, mathematical derivation of the spectral subtraction pre-stage with dynamic noise tracking, over-subtraction factor $\alpha$, and musical noise mitigation. | Acoustic Engineers, Hardware Designers |
| [**05 - Live Demo & Parameters**](05_live_demo.md) | Practical manual for executing [`live_demo.py`](file:///Users/nirajrajendranaphade/Programming/sih-drdo/scripts/live_demo.py): terminal user interface (TUI) visualizer, gain scaling parameters, live bypass toggling, real-time WAV recording, and acoustic latency verification protocols. | Field Operators, QA Testers |
| [**06 - Training & Fine-Tuning Pipeline**](06_training_pipeline.md) | Training data generation using Military Audio Dataset (MAD), voice activity detection (VAD) screening against speech cancellation, multi-resolution STFT loss, asymmetric anti-over-suppression loss penalties, and progressive SNR curriculum schedules. | ML Training Engineers |
| [**07 - Raspberry Pi Hardware Setup**](07_hardware_setup.md) | Complete bare-metal hardware implementation guide: GPIO pinouts for dual INMP441 sensors, I2S kernel device tree overlays (`config.txt`), ALSA soundcard setup, single-core CPU pinning, thermal mitigation, and low-latency Bluetooth sink configuration. | Hardware Engineers, System Integrators |
| [**08 - Evaluation & Benchmarks**](08_evaluation_results.md) | Exhaustive empirical benchmark results: PESQ/STOI/SI-SNR across 824 VCTK-DEMAND test pairs and 830 Military Audio Dataset test splits, statistical significance tables, comparison against classical Boll (1979) DSP, and power/latency profiles on Pi 5. | Research Leads, Defence Auditors |

---

## 5. Quick Start

### Prerequisites & Dependencies

The inference engine runs in a lightweight Python 3.10+ environment requiring only NumPy, SoundFile, SoundDevice, and ONNX Runtime (no PyTorch runtime required on the embedded device):

```bash
# Clone repository and enter root directory
git clone https://github.com/knokvik/sih-drdo.git
cd sih-drdo

# Create and activate an isolated virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install runtime dependencies
pip install -r requirements.txt
```

### Running the Live Dual-Microphone Demonstration

To launch the real-time processing pipeline with the dual INMP441 microphone array, live spectral visualizer, reference mic gain attenuation, and live recording capture:

```bash
python scripts/live_demo.py \
  --dual-mic \
  --native-48k \
  --input-device 1 \
  --visual \
  --ref-gain 0.3 \
  --record-live output.wav
```

### Command Parameter Descriptions

- `--dual-mic`: Activates the dual-sensor acquisition pipeline. Channel 0 is routed as the primary speech channel; Channel 1 is processed as the acoustic reference channel to drive real-time spectral subtraction.
- `--native-48k`: Configures the hardware audio driver to capture at 48,000 Hz natively (required by hardware drivers that reject 16,000 Hz clock rates) and performs high-quality decimation to 16,000 Hz in software.
- `--input-device 1`: Specifies the ALSA/PortAudio device index for the I2S microphone capture interface (use `python scripts/live_demo.py --list-devices` to discover indices).
- `--visual`: Renders an interactive, real-time dual-band spectrogram and volume VU meter directly in the terminal console showing raw input vs. enhanced output spectra.
- `--ref-gain 0.3`: Applies a digital scaling multiplier of $0.3$ to the reference microphone input to compensate for microphone sensitivity disparities and prevent speech over-subtraction.
- `--record-live output.wav`: Streams and persists the clean reconstructed audio directly to disk for post-mission forensic inspection and quality auditing.

### Interactive Live Controls

While the demo script is running, the terminal accepts instantaneous interactive keystrokes:
- Press <kbd>e</kbd>: Toggles the speech enhancement model **ON** (enhanced) or **OFF** (bypass). In bypass mode, the neural network state continues updating in the background to prevent transient phase pops upon re-engagement.
- Press <kbd>q</kbd>: Safely halts the audio stream, flushes synthesis buffers, finalizes WAV file headers, and exits cleanly.

---

## 6. Repository Structure

```
sih-drdo/
├── LICENSE                                # MIT Open Source License
├── README.md                              # High-level project summary and reproduction benchmarks
├── requirements.txt                       # Core Python dependencies (NumPy, SoundFile, ONNX Runtime)
├── archive/                               # Architectural planning drafts and historical notes
│   ├── ARCHITECTURE.md                    # Initial prototype architecture specification
│   └── PLAN.md                            # High-level project milestones and design notes
├── docs/                                  # Project documentation and architectural diagrams
│   ├── SIH2026_PPT_Content.docx           # Presentation briefing document
│   ├── architecture.md                    # High-level architectural briefing and team guide
│   ├── architecture_detailed.png / .svg   # Detailed component-level architectural block diagrams
│   ├── architecture_diagram.png / .svg    # System dataflow schematics
│   ├── demo_script.md                     # Scripted runbook for live jury demonstrations
│   ├── detailed/                          # Complete technical documentation suite (this directory)
│   │   ├── README.md                      # Documentation index & technical overview (this file)
│   │   ├── 01_system_overview.md          # End-to-end tactical system specifications
│   │   ├── 02_model_architecture.md       # GTCRN mathematical architecture & tensor mechanics
│   │   ├── 03_streaming_engine.md         # Low-latency streaming engine & state handling
│   │   ├── 04_dual_mic_system.md          # Dual INMP441 microphone array & DSP pre-stage
│   │   ├── 05_live_demo.md                # Interactive demonstration harness & CLI options
│   │   ├── 06_training_pipeline.md        # Fine-tuning loop, dataset generation, & loss design
│   │   ├── 07_hardware_setup.md           # Raspberry Pi 4/5 hardware guide & ALSA setup
│   │   └── 08_evaluation_results.md       # Empirical PESQ/STOI/SI-SNR benchmark reports
│   ├── eraser_architecture.png            # Visual architecture flow render
│   ├── eraser_workflow.png                # Operational lifecycle diagram
│   ├── learning_path.md                   # Onboarding syllabus for audio DSP and deep learning
│   ├── pipeline_architecture.png / .svg   # Detailed audio pipeline diagrams
│   ├── pipeline_diagram.png / .svg        # Signal flow diagrams
│   ├── ppt_content.md                     # Presentation slide outline and speaker notes
│   ├── references.md                      # Academic literature bibliography and citations
│   ├── references_slide.png / .pptx / .svg# Formatted presentation bibliography slide
│   ├── research-notes.md                  # Comprehensive academic survey and literature analysis
│   ├── solution-design.md                 # Foundational engineering specification and roadmap
│   ├── system_architecture.png / .svg     # System topology diagram
│   └── system_architecture_simple.png/.svg# Simplified high-level architecture diagram
├── scripts/                               # Core operational code, training, and evaluation scripts
│   ├── esc50_noise.py                     # Environmental noise dataset extraction utilities
│   ├── eval_impulsive_noise.py            # Stratified evaluation harness for real defence audio
│   ├── export_onnx.py                     # Causal PyTorch-to-ONNX streaming graph exporter
│   ├── finetune.py                        # Deep neural fine-tuning loop with periodic checkpointing
│   ├── live_demo.py                       # Live audio acquisition, streaming processing, & visualization
│   ├── losses.py                          # Multi-resolution STFT & asymmetric suppression loss modules
│   ├── mad_noise.py                       # Military Audio Dataset (MAD) categorization & ingestion
│   ├── make_demo_noise.py                 # Controlled noise synthesis and mixing scripts
│   ├── make_demo_package.py               # Packager for self-contained demonstration bundles
│   ├── make_demo_samples.py               # Offline batch generation of evaluation audio samples
│   ├── make_demo_video.py                 # Visualization video generation with synchronized waveforms
│   ├── make_film.py                       # Multimedia documentation rendering tool
│   ├── make_pipeline_animation.py         # Frame-by-frame visualizer for neural activations
│   ├── make_pipeline_diagram.py           # Automated rendering tool for pipeline vector graphs
│   ├── make_references_pptx.py            # Automated PowerPoint slide compiler
│   ├── make_references_slide.py           # Slide generator for formal academic references
│   ├── run_baseline_eval.py               # VCTK-DEMAND benchmark reproduction scoring harness
│   ├── screen_noise_speech.py             # Voice Activity Detection (VAD) noise screening tool
│   ├── setup_pi.sh                        # Automated Raspberry Pi OS provisioning & benchmark script
│   ├── spectral_subtraction.py            # Classical Boll (1979) spectral subtraction reference DSP
│   ├── streaming_engine.py                # Frame-by-frame causal streaming inference engine
│   ├── train_dataset.py                   # On-the-fly dynamic speech mixing & augmentation dataset
│   └── verify_onnx_provenance.py          # Cryptographic and numerical graph verification tool
└── third_party/                           # Upstream vendored open-source dependencies
    └── gtcrn/                             # Official GTCRN implementation (ICASSP 2024, MIT License)
        ├── LICENSE                        # Upstream MIT License
        ├── README.md                      # Upstream repository documentation
        ├── gtcrn.py                       # Core PyTorch model definition
        ├── infer.py                       # Batch audio inference utility
        ├── loss.py                        # Original complex STFT loss formulation
        ├── requirements.txt               # Upstream dependency requirements
        ├── checkpoints/                   # Pre-trained model weights (DNS3 and VCTK)
        │   ├── model_trained_on_dns3.tar  # Trained on Microsoft DNS Challenge 3
        │   └── model_trained_on_vctk.tar  # Trained on VoiceBank-DEMAND corpus
        ├── ladspa/                        # C/Rust LADSPA realtime audio plugin implementation
        │   ├── Cargo.toml                 # Rust crate configuration
        │   ├── Dockerfile.minimal-ort     # Minimal ONNX Runtime cross-compilation environment
        │   └── build.sh                   # Build automation script
        ├── stream/                        # Frame-level streaming implementation
        │   ├── README.md                  # Streaming runtime notes
        │   ├── gtcrn.py                   # PyTorch streaming model with manual recurrent caches
        │   ├── gtcrn_stream.py            # PyTorch streaming verification harness
        │   └── onnx_models/               # Pre-exported streaming ONNX computation graphs
        │       └── gtcrn_simple.onnx      # Quantized/simplified streaming ONNX model
        └── test_wavs/                     # Standard benchmark audio verification files
            ├── enh.wav                    # Reference enhanced speech output
            └── mix.wav                    # Reference noisy input speech mixture
```

---

## 7. License and Attribution

This project is licensed under the **MIT License**.

```text
MIT License

Copyright (c) 2024-2026 SIH-DRDO Speech Enhancement Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Academic & Third-Party Attribution

1. **GTCRN Architecture & Checkpoints:**
   - Xiaobin Rong, Yang Sun, Buye Zhang, Yanhua Hu, Daiyong Zhu, and Xianjun Lu. *"GTCRN: A Speech Enhancement Model Requiring Ultra-Low Computational Resources"*, IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024.
   - Codebase vendored under the MIT License from [`Xiaobin-Rong/gtcrn`](https://github.com/Xiaobin-Rong/gtcrn) at commit `502ebfa`.
2. **Military Audio Dataset (MAD):**
   - June-Woo Kim, Sang-Min Yoon, and Ho-Young Jung. *"Military Audio Dataset for Tactical Environmental Sound Classification and Enhancement"*, *Nature Scientific Data* 11:668, 2024. DOI: 10.1038/s41597-024-03516-w.
   - Utilized under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
3. **Classical DSP Benchmark:**
   - Steven F. Boll. *"Suppression of Acoustic Noise in Speech Using Spectral Subtraction"*, IEEE Transactions on Acoustics, Speech, and Signal Processing (TASSP), 27(2):113–120, 1979.
