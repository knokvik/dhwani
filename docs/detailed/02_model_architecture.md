# GTCRN Model Architecture

This document provides an exhaustive technical overview of the Grouped Temporal Convolutional Recurrent Network (GTCRN), the core speech enhancement model utilized in this project. The architecture is based on the ICASSP 2024 publication by Rong, Sun, Zhang, Hu, Zhu, and Lu.

## 1. Overview

**GTCRN** stands for **Grouped Temporal Convolutional Recurrent Network**. It is an extremely lightweight, real-time speech enhancement model designed for efficient processing of complex Short-Time Fourier Transform (STFT) spectrograms.

- **Publication:** ICASSP 2024 (Rong, Sun, Zhang, Hu, Zhu & Lu)
- **Parameters:** 23.67K learnable parameters (48.2K total, including fixed Equivalent Rectangular Bandwidth (ERB) filterbank parameters)
- **Computational Cost:** 33.0 MMACs/s (Million Multiply-Accumulate Operations per second)
- **Input Type:** Complex STFT spectrograms

The architecture employs advanced techniques such as Grouped Temporal Convolutions, Channel Shuffling, and Dual-Path Grouped RNNs to drastically reduce the parameter count and computational footprint without sacrificing perceptual quality.

---

## 2. Input Representation

The input to the model relies on the STFT of the raw audio waveform, with carefully chosen parameters to balance frequency resolution and temporal latency for real-time applications.

- **Sample Rate:** 16,000 Hz
- **N_FFT (FFT Size):** 512 points (corresponding to a 32 ms window)
- **Hop Length:** 256 points (16 ms hop size, yielding 50% overlap)
- **Window Function:** Periodic square-root Hann (sqrt-Hann) window, defined as:

$$ w[n] = \sqrt{0.5 - 0.5 \cos\left(\frac{2\pi n}{N_{FFT}}\right)} $$

The STFT yields a complex spectrum. Taking the positive frequencies and DC (since the time-domain signal is real), the number of frequency bins is $N_{FFT} / 2 + 1 = 257$.

For $B$ batch size and $T$ time frames, the initial shape of the complex STFT output is `(B, 257, T, 2)` where `2` represents the real and imaginary components.

The model is fed a 3-channel stack representing the magnitude, real, and imaginary parts:
1. $X_{\text{mag}} = \sqrt{X_{\text{real}}^2 + X_{\text{imag}}^2}$
2. $X_{\text{real}}$
3. $X_{\text{imag}}$

This results in an input tensor of shape **`(B, 3, T, 257)`**, which is subsequently transposed or processed along the frequency dimension.

---

## 3. Equivalent Rectangular Bandwidth (ERB) Compression

Human auditory perception exhibits logarithmic frequency resolution—we perceive fine spectral details at low frequencies, but our resolution broadens at higher frequencies. GTCRN utilizes an ERB-based filterbank to compress the 257 linear frequency bins into a psychoacoustically informed lower-dimensional representation.

- **Low Frequencies (0 - 2 kHz):** 65 bins are preserved linearly.
- **High Frequencies (2 - 8 kHz):** 192 bins are compressed into 64 ERB bands using a triangular overlap-add filterbank.
- **Total Compressed Bins:** $65 + 64 = 129$ bins.

The compression matrix `erb_fc` is **fixed and non-learnable**. It is purely derived from the psychoacoustic ERB scale. The inverse transformation `ierb_fc` reconstructs the 129 bins back to 257 via transpose filtering.

By reducing the frequency dimension from 257 to 129, the ERB compression reduces the overall computational complexity by approximately 50% while maintaining perceptual quality.

---

## 4. Subband Feature Extraction (SFE)

To capture local spectral variations (spectral slope), the model employs a Subband Feature Extraction mechanism.

This is implemented using PyTorch's `nn.Unfold` over the frequency axis:
- **Kernel Size:** `(1, 3)` (1 frame in time, 3 bins in frequency)
- **Padding:** `(0, 1)`

For each of the 3 input channels (magnitude, real, imaginary), SFE aggregates the target frequency bin and its two immediate neighbors.
- **Output Channels:** $3 \times 3 = 9$ channels.

This operation allows the encoder to perceive local spectral context without introducing any learnable parameters.

---

## 5. Encoder Architecture

The encoder progressively downsamples the frequency resolution while expanding the channel capacity. It relies on standard convolutional blocks and the specialized GTConvBlocks.

1. **Layer 1:** ConvBlock
   - Mapping: 9 $\to$ 16 channels
   - Kernel: `(1, 5)`, Stride: `(1, 2)`, Padding: `(0, 2)`
   - Frequency Reduction: $129 \to 65$
2. **Layer 2:** ConvBlock
   - Mapping: 16 $\to$ 16 channels
   - Kernel: `(1, 5)`, Stride: `(1, 2)`, Padding: `(0, 2)`, Groups: 2
   - Frequency Reduction: $65 \to 33$
3. **Layers 3, 4, 5:** GTConvBlock (Grouped Temporal Convolution Block)
   - Kernel: `(3, 3)`
   - Temporal Dilations: $d = 1, 2, 5$
   - Note: Each ConvBlock consists of a `Conv2d`, `BatchNorm`, and `PReLU` activation.

---

## 6. GTConvBlock (Group Temporal Convolution Block)

The **GTConvBlock** is the key innovation in this architecture, heavily inspired by the highly efficient **ShuffleNetV2** design (Ma et al., 2018). It drastically cuts parameters by processing only a fraction of the channels.

Given an input tensor with 16 channels, the channels are split in half:
- **Branch $x_1$:** 8 channels
- **Branch $x_2$:** 8 channels (Identity pass-through)

### Branch $x_1$ Processing:
1. **Subband Feature Extraction (SFE):** Unfold 8 channels $\to$ $8 \times 3 = 24$ channels.
2. **Pointwise Conv:** $1 \times 1$ convolution reducing 24 channels back to 16, followed by `BatchNorm` + `PReLU`.
3. **Causal Depthwise Conv2d:** $3 \times 3$ kernel, grouped into 16 channels, with temporal dilation $(d, 1)$, followed by `BatchNorm` + `PReLU`. It enforces causality (no future lookahead).
4. **Pointwise Conv:** $1 \times 1$ convolution reducing 16 channels $\to$ 8 channels, followed by `BatchNorm`.
5. **Temporal Recurrent Attention (TRA):**
   - Energy pooling across the frequency dimension.
   - Passes through a single-layer GRU (hidden size = 8).
   - Linear projection followed by a Sigmoid activation to generate a channel-wise gate $A_t \in [0, 1]$.
   - The output $h_1$ is gated: $h_1 = h_1 \odot A_t$.

### Recombination:
- **Branch $x_2$:** Undergoes zero computation.
- **Channel Shuffle:** The outputs $h_1$ and $x_2$ are concatenated and their channels are interleaved (shuffled) to ensure cross-branch information flow in subsequent layers.

By forcing half the channels to skip computation entirely and applying grouped/depthwise convolutions to the active half, GTConvBlock achieves profound efficiency.

---

## 7. DPGRNN (Dual-Path Grouped RNN)

Temporal and harmonic patterns are modeled at the bottleneck using two cascaded DPGRNN layers. The Dual-Path architecture separates the modeling of intra-frame (frequency) and inter-frame (time) dynamics.

### 7.1. Intra-RNN (Frequency Modeling)
Models harmonic structures across the frequency axis within a single time frame.
- **Reshape:** To `(B*T, F, C)` where $F = 33$ frequency positions.
- **RNN:** Bidirectional GRNN with hidden size = 8 for each direction (total 16).
- **Post-processing:** Linear layer $\to$ LayerNorm $\to$ Residual connection.

### 7.2. Inter-RNN (Temporal Modeling)
Models temporal dynamics across consecutive time frames for a given frequency band.
- **Reshape:** To `(B*F, T, C)` where $F = 33$.
- **RNN:** Unidirectional **CAUSAL** GRNN with hidden size = 16. (Causal to preserve real-time streaming constraints).
- **Post-processing:** Linear layer $\to$ LayerNorm $\to$ Residual connection.

### What is a Grouped RNN (GRNN)?
Instead of a single large RNN, GRNN splits the input channels into 2 groups, feeding each group into an independent smaller GRU. This halves the parameter count for the RNN layers.

---

## 8. Decoder Architecture

The decoder mirrors the encoder, utilizing transposed convolutions for upsampling, and employing skip connections to recover fine-grained details lost during encoding.

- **GTConvBlocks:** 3 layers of GTConvBlock (with `use_deconv=True`), using dilations $d=5, 2, 1$.
- **ConvTranspose2d:** 2 layers upsampling the frequency resolution: $33 \to 65 \to 129$.
- **Skip Connections:** Added symmetrically: $X_{dec} = X_{dec} + \text{EncoderOutput}_{N-1-i}$.

---

## 9. Complex Ratio Mask (CRM)

The output of the decoder is a 2-channel tensor representing a complex mask:
- $M_r$: Real Mask
- $M_i$: Imaginary Mask

The enhanced complex spectrum $\hat{S}$ is obtained by applying the complex mask to the input complex spectrum $S_{in} = X_r + j X_i$:

$$ \hat{S}_r = X_r M_r - X_i M_i $$
$$ \hat{S}_i = X_i M_r + X_r M_i $$

This represents a true complex multiplication $\hat{S} = S_{in} \times M$.
Unlike Ideal Ratio Masks (IRM) which only scale the magnitude, CRM rotates the phase and scales the magnitude simultaneously. This drastically reduces phase-discontinuity artifacts, such as "musical noise".

---

## 10. Streaming State Management

To operate in real-time on a frame-by-frame basis, GTCRN maintains explicit internal memory (cache tensors) to carry context from previous frames.

- **`conv_cache`**: Shape `(2, 1, 16, 16, 33)`. Stores receptive field buffers for the causal temporal convolutions in the encoder and decoder.
- **`tra_cache`**: Shape `(2, 3, 1, 1, 16)`. Stores the GRU hidden states for the 3 Temporal Recurrent Attention (TRA) units in the encoder and decoder.
- **`inter_cache`**: Shape `(2, 1, 33, 16)`. Stores the GRU hidden states for the Inter-RNN across the 33 frequency channels for the 2 DPGRNN layers.

The total memory footprint for these caches is roughly ~10 KB, which is negligible, making it highly suitable for embedded edge devices.

---

## 11. Architecture Summary and Parameter Breakdown

The following table breaks down the parameter counts for each module.

| Component | Parameters | Learnable |
| :--- | :--- | :---: |
| ERB Filterbank | 24,576 | ❌ |
| Encoder Conv Layers | ~4,000 | ✅ |
| Encoder GTConvBlocks | ~8,000 | ✅ |
| DPGRNN Layers | ~6,000 | ✅ |
| Decoder GTConvBlocks | ~4,000 | ✅ |
| Decoder Conv Layers | ~1,600 | ✅ |
| Complex Mask Layer | ~100 | ✅ |
| **Total Learnable** | **23,670** | |
| **Total (inc. fixed)**| **48,246** | |

### Complete Architecture Data Flow Diagram

```mermaid
flowchart TD
    %% Inputs
    AudioIn[Raw Audio Waveform] --> STFT[STFT\nN_FFT=512, Hop=256]
    STFT --> |Complex Spectrum\nB, 257, T, 2| Split[Split into Mag, Real, Imag]
    Split --> Stack[Stack Channels\nB, 3, T, 257]
    
    %% ERB
    Stack --> ERB[ERB Compression Matrix\nFixed 257 -> 129]
    ERB --> SFE[Subband Feature Extraction\nnn.Unfold (1,3)]
    
    %% Encoder
    SFE --> |B, 9, T, 129| EncConv1[Encoder Conv 1\nFreq: 129 -> 65\nChan: 9 -> 16]
    EncConv1 --> EncConv2[Encoder Conv 2\nFreq: 65 -> 33\nChan: 16 -> 16]
    EncConv2 --> GTB_E1[GTConvBlock d=1]
    GTB_E1 --> GTB_E2[GTConvBlock d=2]
    GTB_E2 --> GTB_E3[GTConvBlock d=5]
    
    %% Bottleneck
    GTB_E3 --> DPGRNN[2x DPGRNN\nDual-Path Grouped RNN]
    
    %% Decoder
    DPGRNN --> GTB_D1[Decoder GTConvBlock d=5\n+ Skip from E3]
    GTB_E2 --> |Skip Connection| GTB_D1
    GTB_D1 --> GTB_D2[Decoder GTConvBlock d=2\n+ Skip from E2]
    GTB_E1 --> |Skip Connection| GTB_D2
    GTB_D2 --> GTB_D3[Decoder GTConvBlock d=1\n+ Skip from E1]
    
    GTB_D3 --> DecConv1[Decoder ConvTranspose 1\nFreq: 33 -> 65]
    EncConv2 --> |Skip Connection| DecConv1
    DecConv1 --> DecConv2[Decoder ConvTranspose 2\nFreq: 65 -> 129]
    EncConv1 --> |Skip Connection| DecConv2
    
    %% Masking and Output
    DecConv2 --> MaskProj[Mask Projection\n2 Channels: M_r, M_i]
    MaskProj --> InverseERB[Inverse ERB Expansion\nFixed 129 -> 257]
    
    InverseERB --> ComplexMul[Complex Multiplication\nApply Mask to Original Spectrum]
    STFT --> |Delayed Original Spectrum| ComplexMul
    
    ComplexMul --> iSTFT[Inverse STFT\nReconstruct Waveform]
    iSTFT --> AudioOut[Enhanced Audio Waveform]
```
