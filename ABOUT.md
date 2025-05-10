# WaveSplit Audio Denoiser: Complete Project Analysis

I'll provide an in-depth analysis of your WaveSplit Audio Denoiser project, explaining how the system works step-by-step, the algorithms used, and the benefits of your enhanced approach compared to existing methods.

## 1. Project Overview

WaveSplit is an advanced audio denoising system built on CleanUNet with significant enhancements for improved audio quality. The project combines several novel techniques:

1. Enhanced neural network-based denoising (base CleanUNet model)
2. Adaptive SNR-based processing
3. Harmonic enhancement for clearer speech
4. Vocal clarity enhancement focusing on speech frequencies 
5. Perceptual enhancement based on psychoacoustic principles
6. Optional dynamic range compression
7. Comprehensive metrics and visualization tools

The system is packaged in a user-friendly Gradio interface that allows users to upload or record audio and process it with various enhancement options.

## 2. System Architecture

### 2.1 Core Components and File Structure

The project consists of several Python modules with specific responsibilities:

- `gradio_app.py`: Main application entry point with the Gradio UI
- `enhanced_denoiser.py`: Contains the enhanced denoising engine (`EnhancedDenoiserAudio` class)
- `denoiser.py`: Contains the base denoising engine (`DenoiserAudio` class)
- `visualization_utils.py`: Tools for audio analysis, metrics calculation, and visualization
- `advanced_metrics.py`: Implementation of sophisticated audio quality metrics
- `model_comparison.py`: Tools for comparing the enhanced model with baseline models
- `utils.py`: Helper functions for audio chunking and processing
- `__init__.py`: Package definition exposing the main classes

### 2.2 Base Technologies

The project builds upon several key technologies:

1. **CleanUNet**: A neural network architecture for audio denoising
2. **PyTorch/TorchAudio**: For deep learning and audio processing
3. **Librosa**: For audio analysis and feature extraction
4. **Gradio**: For the interactive web interface
5. **Matplotlib**: For visualization of audio data and metrics
6. **SciPy**: For signal processing operations

## 3. Processing Flow

When a user processes an audio file through the WaveSplit system, the following steps occur:

### 3.1 Initial Processing and Audio Loading

1. The user uploads or records an audio file through the Gradio interface
2. The interface calls the `process_audio()` function with the audio input and enhancement options
3. Inside `process_audio()`, the system:
   - Creates temporary directories for output and metrics
   - Checks if the input is a recorded audio (tuple of sample rate and data) or an uploaded file
   - Loads and normalizes the audio data

### 3.2 Core Denoising Process

4. The audio is then passed to the `EnhancedDenoiserAudio` instance (`denoise` variable)
5. The `__call__` method of `EnhancedDenoiserAudio` processes the audio:
   - Loads and resamples the audio (ensuring 16kHz sampling rate)
   - Analyzes the original audio to extract metrics and classify noise types
   - Chunks the long audio into smaller segments (by default 3-second chunks)
   - Processes each chunk through the neural network model (CleanUNet)
   - Applies post-processing enhancements based on user settings

### 3.3 Neural Network Processing

6. The chunking phase uses `chunk_audio()` from `utils.py`:
   - Divides audio into fixed-length segments
   - Handles padding for the last chunk if needed
   - Returns a tensor of chunks

7. The neural network processing occurs in the `denoise()` method:
   - Divides chunks into batches (default max 20 chunks per batch)
   - Processes each batch through the CleanUNet model
   - Applies adaptive processing if enabled, adjusting the processing strategy based on the estimated SNR
   - Optionally applies domain adaptation (currently a placeholder feature)

8. After denoising, the chunks are stitched back together using `unchunk_audio()` from `utils.py`:
   - Combines chunks with proper alignment
   - Handles potential overlapping to avoid discontinuities

### 3.4 Audio Enhancement Stages

9. After neural network processing, the audio undergoes several enhancement stages:

   a. **Perceptual Enhancement** (always active):
      - Uses psychoacoustic principles to enhance perceptual qualities
      - Applies frequency-dependent weighting based on human hearing sensitivity
      - Implemented in `perceptual_enhancement_filter()`

   b. **Harmonic Enhancement** (if enabled):
      - Separates harmonic components from percussive/noise components
      - Enhances harmonic content to improve speech clarity
      - Implemented in `harmonic_enhancement_filter()`

   c. **Vocal Clarity Enhancement** (if enabled):
      - Applies bandpass filtering focused on speech frequencies (300-3000 Hz)
      - Blends with original to maintain natural sound qualities
      - Implemented in `vocal_clarity_enhancement()`

   d. **Dynamic Range Compression** (if enabled):
      - Reduces the difference between loud and soft sounds
      - Makes speech more consistent in volume
      - Implemented in `apply_dynamic_range_compression()`

### 3.5 Metrics Calculation and Visualization

10. After processing, the system calculates comprehensive metrics:
    - Compares original and processed audio
    - Extracts metrics like SNR improvement, spectral balance, dynamic range
    - Classifies noise types present in the original audio
    - Compiles everything into a metrics dictionary

11. Visualization tools in `AudioMetrics` class generate:
    - Spectrograms comparing original and denoised audio
    - Waveform visualizations
    - Power spectral density (PSD) comparisons
    - Noise reduction maps
    - Frequency band energy distributions

12. The `ModelComparison` class provides:
    - Radar charts comparing WaveSplit with baseline models
    - Bar charts for objective metrics
    - Processing time charts
    - LaTeX tables for research paper inclusion

### 3.6 Results Presentation

13. The Gradio interface displays:
    - The denoised audio for playback
    - Spectrogram visualizations
    - Detailed metrics in an HTML report
    - Additional comparison charts
    - A summary of which enhancements were applied

## 4. Key Algorithms and Techniques Explained

### 4.1 CleanUNet Neural Network

At the core of WaveSplit is CleanUNet, a deep learning model specifically designed for audio denoising. CleanUNet combines:

1. **U-Net Architecture**: An encoder-decoder structure that captures multi-scale features. The encoder compresses audio into a latent representation, while the decoder reconstructs clean audio from this representation.

2. **Convolutional Layers**: Process audio in the time domain, learning to separate noise from speech.

3. **Skip Connections**: Connect corresponding layers in the encoder and decoder, helping preserve detailed information that might be lost during encoding.

The model is pre-trained on diverse audio datasets to learn the difference between speech and various noise types.

### 4.2 Adaptive SNR-based Processing

The adaptive processing feature analyzes the Signal-to-Noise Ratio (SNR) of audio chunks and adjusts the processing strategy accordingly:

1. **SNR Estimation**: Implemented in `estimate_snr()`, which:
   - Calculates the signal envelope
   - Estimates noise floor using lower percentiles
   - Estimates signal level using higher percentiles
   - Calculates SNR as the ratio between signal and noise levels in dB

2. **Processing Strategy Adaptation**:
   - For high SNR (>30dB) segments, applies "light" processing to preserve more original details
   - For standard SNR segments, applies full denoising
   - Blends original and denoised audio in appropriate ratios depending on SNR

This approach prevents over-processing of already clean segments, maintaining natural sound quality.

### 4.3 Harmonic Enhancement

Speech consists primarily of harmonic components, while noise is often non-harmonic. The harmonic enhancement technique:

1. Uses **Harmonic-Percussive Source Separation (HPSS)** to separate audio components:
   - Harmonic components have horizontal structure in the spectrogram (sustained tones)
   - Percussive components have vertical structure (transients)

2. Enhances harmonic components which typically contain speech:
   - Blends the original audio with enhanced harmonic components
   - Uses a 70/30 blend to maintain balance

This technique improves speech intelligibility by emphasizing the harmonic structure of speech while reducing noise.

### 4.4 Vocal Clarity Enhancement

The vocal clarity enhancement focuses on the frequency range most important for speech intelligibility:

1. Applies a **bandpass filter** to focus on 300-3000 Hz range:
   - Designs a Butterworth filter with specified cutoffs
   - Uses bidirectional filtering (filtfilt) to prevent phase distortion

2. Blends filtered audio with original:
   - Uses a 70/30 blend (70% original, 30% filtered)
   - Maintains natural sound qualities while enhancing speech frequencies

This technique improves the perception of consonants and vowels without making the audio sound artificial.

### 4.5 Perceptual Enhancement

The perceptual enhancement filter applies psychoacoustic principles to improve perceived audio quality:

1. **Equal-Loudness Contour Approximation**:
   - Models the non-linear frequency sensitivity of human hearing
   - Applies weights to different frequency bands based on this model

2. **Short-Time Fourier Transform (STFT)** processing:
   - Converts audio to time-frequency domain
   - Applies perceptual weighting to magnitude components
   - Reconstructs signal with enhanced magnitudes and original phase

This produces audio that sounds better to human ears by emphasizing frequencies we're more sensitive to while de-emphasizing less perceptible frequencies.

### 4.6 Dynamic Range Compression

Dynamic range compression reduces the volume difference between loud and soft sounds:

1. **Level detection**:
   - Converts amplitude to dB scale
   - Identifies portions above a threshold

2. **Compression**:
   - Reduces levels above threshold according to compression ratio
   - Maintains levels below threshold
   - Converts back to linear scale

3. **Normalization**:
   - Adjusts output level to match input peak level
   - Prevents distortion from clipping

This makes speech more consistently audible, especially in noisy environments or on devices with limited dynamic range.

### 4.7 Noise Classification

The system analyzes noise characteristics to identify the type of noise present:

1. **Feature extraction**:
   - Spectral centroid (brightness)
   - Zero crossing rate (noisiness)
   - Spectral bandwidth (spread)
   - Spectral rolloff (high frequency content)

2. **Classification logic**:
   - White noise: high ZCR, high centroid
   - Pink noise: medium ZCR, medium centroid
   - Street noise: medium ZCR, variable bandwidth
   - Crowd noise: low-medium ZCR, high spectral variability
   - Mechanical noise: low ZCR, specific rolloff pattern
   - Office noise: very low ZCR, low spectral centroid

3. **Probability assignment**:
   - Assigns likelihood scores to each noise type
   - Normalizes to create a probability distribution

This information helps users understand their noise environment and can guide enhancement strategies.

### 4.8 Audio Chunking and Stitching

Processing long audio files efficiently requires chunking and stitching:

1. **Chunking** (`chunk_audio` function):
   - Divides long audio into fixed-length segments (default 3 seconds)
   - Handles padding for incomplete chunks
   - Creates a tensor of chunks for batch processing

2. **Stitching** (`unchunk_audio` function):
   - Recombines processed chunks into a continuous output
   - Handles dimension reshaping for compatibility
   - Applies tapering for smooth transitions between chunks (if overlap is used)

This approach enables processing arbitrarily long audio files while managing memory efficiently.

## 5. Metrics and Evaluation

WaveSplit implements comprehensive metrics to evaluate audio quality:

### 5.1 Basic Audio Metrics

1. **Signal-to-Noise Ratio (SNR)**:
   - Measures the ratio between signal power and noise power in dB
   - Higher values indicate better noise reduction
   - Calculated using signal statistics and energy ratios

2. **Spectral Centroid**:
   - Measures the "center of mass" of the spectrum
   - Indicates the brightness or darkness of a sound
   - Helps evaluate if speech tone is maintained

3. **Zero Crossing Rate**:
   - Counts how often the signal changes sign
   - Higher values typically indicate more noise
   - Used to evaluate noise reduction

4. **Dynamic Range**:
   - Measures the difference between the loudest and quietest parts
   - Important for evaluating compression effects

### 5.2 Advanced Audio Metrics

1. **Harmonic-to-Noise Ratio (HNR)**:
   - Measures the ratio between harmonic and noise components
   - Higher values indicate clearer speech
   - Uses pitch tracking and harmonic isolation techniques

2. **Segmental SNR**:
   - Calculates SNR for short time segments and averages
   - More perceptually relevant than global SNR
   - Handles variable noise levels

3. **Log-Spectral Distance (LSD)**:
   - Measures the difference between spectrograms in the log domain
   - Lower values indicate better preservation of spectral characteristics
   - Sensitive to both amplitude and frequency changes

4. **Speech Intelligibility**:
   - Estimates how well speech can be understood
   - Uses correlation in speech-important frequency bands
   - Focuses on 500-4000 Hz range

### 5.3 Perceptual Evaluation Metrics

The system also implements standard perceptual metrics:

1. **Perceptual Evaluation of Speech Quality (PESQ)**:
   - Industry standard for evaluating speech quality
   - Produces a score between 1 (bad) and 5 (excellent)
   - Models human perception of audio quality

2. **Short-Time Objective Intelligibility (STOI)**:
   - Measures speech intelligibility
   - Produces a score between 0 and 1
   - Correlates well with human listening tests

3. **Speech Distortion Index (SDI)**:
   - Measures the amount of distortion introduced by processing
   - Lower values indicate less distortion
   - Helps evaluate if the denoising process is causing artifacts

## 6. User Interface and Experience

The Gradio interface provides a user-friendly way to interact with the system:

### 6.1 Main Interface Components

1. **Upload Audio Tab**:
   - Allows users to upload audio files
   - Provides enhancement settings with checkboxes
   - Shows processing status and results

2. **Record Audio Tab**:
   - Enables direct recording from microphone
   - Has the same enhancement options
   - Displays processing results

3. **Results Display**:
   - Shows denoised audio for playback
   - Provides tabbed view of metrics and visualizations
   - Includes spectrograms, waveforms, and PSD plots

### 6.2 Enhancement Settings

Users can toggle four key enhancement features:

1. **Adaptive SNR Processing**:
   - Dynamically adjusts processing based on noise levels
   - On by default

2. **Harmonic Enhancement**:
   - Enhances speech harmonics for better clarity
   - On by default

3. **Vocal Clarity**:
   - Enhances frequency bands important for speech
   - On by default

4. **Dynamic Range Compression**:
   - Reduces dynamic range for more consistent volume
   - Off by default

The interface notes that "Perceptual Enhancement" is always active and cannot be disabled.

### 6.3 Visual Feedback

The interface provides rich visual feedback:

1. **Spectrograms**:
   - Shows time-frequency representations of original and denoised audio
   - Helps visualize noise reduction and preservation of speech components

2. **Waveform Comparison**:
   - Displays amplitude vs. time for original and denoised audio
   - Shows how the processing affects the signal shape

3. **Power Spectral Density**:
   - Shows the distribution of power across frequencies
   - Helpful for evaluating frequency-dependent noise reduction

4. **Noise Reduction Map**:
   - Visualizes which time-frequency regions were most affected by denoising
   - Highlights where noise was removed

## 7. Advantages Over Traditional Methods

WaveSplit offers several advantages over traditional audio denoising approaches:

### 7.1 Compared to Traditional Digital Signal Processing

1. **Adaptivity**: Traditional methods like spectral subtraction or Wiener filtering use fixed parameters, while WaveSplit adapts to the specific noise characteristics of the input.

2. **Non-linear Processing**: Neural networks can learn complex non-linear relationships between noisy and clean audio, outperforming linear filters.

3. **Contextual Understanding**: WaveSplit considers the temporal context of audio, not just isolated frames, leading to more coherent results.

4. **Perceptual Optimization**: The system is optimized for human perception rather than mathematical criteria alone.

### 7.2 Compared to Basic Neural Networks

1. **Enhanced Post-processing**: WaveSplit adds sophisticated post-processing steps that address limitations of neural network output.

2. **Harmonic Preservation**: The harmonic enhancement specifically preserves speech components that might be damaged by aggressive denoising.

3. **Adaptive Processing**: The SNR-based adaptation prevents over-processing of already clean segments, a common problem in neural denoisers.

4. **Comprehensive Metrics**: The extensive metrics suite allows precise evaluation and tuning of results.

### 7.3 Real-world Benefits

1. **Improved Communication**: By enhancing speech clarity and reducing noise, WaveSplit makes conversations more intelligible, especially in challenging environments.

2. **Reduced Listening Fatigue**: By removing noise and enhancing speech frequencies, the system reduces the cognitive load required to understand speech.

3. **Better Accessibility**: For those with hearing impairments, the vocal clarity enhancements can make a significant difference in speech comprehension.

4. **Enhanced Audio Production**: For podcasts, videos, or other audio content, WaveSplit can salvage recordings made in noisy environments.

## 8. Implementation Details and Optimizations

### 8.1 Performance Optimizations

1. **Batch Processing**:
   - Processes multiple audio chunks simultaneously
   - Efficiently utilizes GPU computation
   - Configurable batch size (default 20 chunks)

2. **Memory Management**:
   - Explicitly frees GPU memory after each batch
   - Prevents out-of-memory errors for long audio files

3. **Efficient Audio Handling**:
   - Uses memory-mapped operations where possible
   - Handles multi-channel audio by converting to mono

### 8.2 Error Handling and Robustness

The code includes comprehensive error handling:

1. **Graceful Degradation**:
   - Falls back to simpler methods if advanced ones fail
   - Returns original audio if processing fails completely

2. **Input Validation**:
   - Checks audio formats and shapes
   - Handles edge cases like very short audio segments

3. **Safe Numerical Operations**:
   - Uses epsilon values to prevent division by zero
   - Applies clamping to prevent extreme values

### 8.3 Modular Design

The system is designed with modularity in mind:

1. **Separation of Concerns**:
   - Base denoiser vs. enhanced denoiser
   - Processing logic vs. UI logic
   - Metrics calculation vs. visualization

2. **Extensibility**:
   - Easy to add new enhancement techniques
   - Support for domain adaptation
   - Framework for model comparison

This design makes it easy to maintain and extend the system with new capabilities.

## 9. Potential Future Enhancements

Based on the codebase, several potential enhancements could be implemented:

1. **Real Domain Adaptation**:
   - Fully implement the placeholder domain adaptation feature
   - Train specialized adapters for specific noise environments
   - Create a library of pre-trained domain adapters

2. **Enhanced Listening Tests**:
   - Complete the listening test tab functionality
   - Implement formal ITU-R BS.1116 and MUSHRA test protocols
   - Add statistical analysis of listening test results

3. **Real-time Processing**:
   - Optimize for lower latency
   - Enable streaming audio processing
   - Develop a standalone application outside the browser

4. **Additional Enhancement Techniques**:
   - More sophisticated harmonic regeneration
   - De-reverberation capabilities
   - Speaker separation for multi-speaker scenarios

## 10. Conclusion

WaveSplit represents a sophisticated audio denoising system that combines the power of deep learning with advanced signal processing techniques. By addressing the limitations of both traditional methods and basic neural network approaches, it achieves superior audio quality and user control.

The system's modular architecture, comprehensive metrics, and intuitive interface make it both powerful and accessible. The attention to psychoacoustic principles ensures that the results not only measure well numerically but also sound good to human listeners.

The combination of adaptive processing, harmonic enhancement, vocal clarity enhancement, and perceptual optimization represents a significant advancement in the field of audio denoising, with practical benefits for communication, content creation, and accessibility.