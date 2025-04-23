from cleanunet import CleanUNet
import torchaudio
import torch
import numpy as np
import librosa
from typing import Union, Optional, Dict, Any, Tuple, List
import os
from tqdm import tqdm
from denoiser.utils import chunk_audio, unchunk_audio
import scipy.signal as signal

class EnhancedDenoiserAudio():
    """
    Enhanced Denoiser module with novel modifications.
    
    Args:
        device (str, Optional): Device to use. Defaults to GPU if available.
        chunk_length_s (int, Optional): Length of a single chunk in second.
        max_batch_size (int, Optional): Maximum size of a batch to infer in one instance.
        adaptive_processing (bool): Whether to use adaptive SNR-based processing.
        harmonic_enhancement (bool): Whether to use harmonic enhancement.
        vocal_clarity (bool): Whether to use vocal clarity enhancement.
        dynamic_range_compression (bool): Whether to apply dynamic range compression.
        domain_adaptation (bool): Whether to use domain-specific adaptation.
        domain_type (str): Type of domain for adaptation ("office", "outdoors", "vehicle", etc).
        verbose (bool): Whether to print verbose output.
    """
    
    def __init__(self,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 chunk_length_s: int = 3,
                 max_batch_size: int = 20,
                 adaptive_processing: bool = True,
                 harmonic_enhancement: bool = True,
                 vocal_clarity: bool = True,
                 dynamic_range_compression: bool = False,
                 domain_adaptation: bool = False,
                 domain_type: str = "general",
                 verbose: bool = True) -> None:
        
        self.device = device
        self.chunk_length_s = chunk_length_s
        self.max_batch_size = max_batch_size
        self.adaptive_processing = adaptive_processing
        self.harmonic_enhancement = harmonic_enhancement
        self.vocal_clarity = vocal_clarity
        self.dynamic_range_compression = dynamic_range_compression
        self.domain_adaptation = domain_adaptation
        self.domain_type = domain_type
        self.verbose = verbose
        
        # Internal parameters for perceptual enhancement - always active
        self._perceptual_enhancement = True
        
        if self.verbose:
            print(f"Initializing EnhancedDenoiserAudio with settings:")
            print(f"- device: {device}")
            print(f"- chunk_length_s: {chunk_length_s}")
            print(f"- max_batch_size: {max_batch_size}")
            print(f"- adaptive_processing: {adaptive_processing}")
            print(f"- harmonic_enhancement: {harmonic_enhancement}")
            print(f"- vocal_clarity: {vocal_clarity}")
            print(f"- dynamic_range_compression: {dynamic_range_compression}")
            print(f"- domain_adaptation: {domain_adaptation}")
            print(f"- domain_type: {domain_type}")
            print(f"- perceptual_enhancement: Always active")
        
        # Load base model
        self.model = CleanUNet.from_pretrained(device=device)
        
        # Initialize domain adapter if needed
        if self.domain_adaptation:
            self.domain_adapter = self._create_domain_adapter().to(device)
            # Here you'd typically load pre-trained weights for the domain adapter
            # self._load_domain_weights(domain_type)
        
    def _create_domain_adapter(self) -> torch.nn.Module:
        """Create a lightweight domain adaptation layer"""
        return DomainAdapter(hidden_dim=64).to(self.device)
    
    def _load_domain_weights(self, domain_type: str) -> None:
        """Load pre-trained weights for a specific domain"""
        # This would be implemented based on available pre-trained adapters
        adapter_path = f"adapters/{domain_type}_adapter.pt"
        if os.path.exists(adapter_path):
            self.domain_adapter.load_state_dict(torch.load(adapter_path))
            if self.verbose:
                print(f"Loaded domain adapter for {domain_type}")
        else:
            if self.verbose:
                print(f"No pre-trained adapter found for {domain_type}, using default")
    
    @staticmethod
    def load_audio_and_resample(audio_path: str, target_sr: int = 16000) -> torch.Tensor:
        """
        Loads audio and resamples to target sampling rate. Returns single channel.

        Args:
            audio_path (str): Path to audio file.
            target_sr (int, optional): Target sampling rate. Defaults to 16000.

        Returns:
            torch.Tensor: Tensor of audio file, returns single channel only.
        """
        try:
            audio_wav, s_rate = torchaudio.load(audio_path)
            
            # If multi-channel, convert to mono
            if audio_wav.shape[0] > 1:
                audio_wav = torch.mean(audio_wav, dim=0, keepdim=True)
                
            # Resample if needed
            if s_rate != target_sr:
                resampler = torchaudio.transforms.Resample(s_rate, target_sr)
                audio_wav = resampler(audio_wav)
                
            return audio_wav[0]  # Return single channel
        except Exception as e:
            raise RuntimeError(f"Error loading audio file: {str(e)}")
    
    @staticmethod
    def audio_processing(audio_wav: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Process audio. Trims all zeros.

        Args:
            audio_wav (torch.Tensor | np.ndarray): Audio wave loaded.

        Raises:
            TypeError: If file is not a Tensor or an Array.

        Returns:
            np.ndarray: Zero trimmed audio array. 
        """
        if isinstance(audio_wav, torch.Tensor):
            # Ensure it's on CPU and convert to numpy
            audio_wav = audio_wav.cpu().detach().numpy()
            return np.trim_zeros(audio_wav)
        elif isinstance(audio_wav, np.ndarray):
            return np.trim_zeros(audio_wav)
        else:
            raise TypeError(f'Only supports numpy.ndarray or torch.Tensor file types. Got {type(audio_wav)}')
    
    @staticmethod
    def estimate_snr(signal: np.ndarray) -> float:
        """
        Estimate signal-to-noise ratio of an audio segment.
        
        Args:
            signal (np.ndarray): Audio signal.
            
        Returns:
            float: Estimated SNR value in dB.
        """
        # Compute signal envelope using Hilbert transform or simple abs
        try:
            # Use analytic signal method if available
            analytic_signal = np.abs(signal)
            envelope = np.abs(analytic_signal)
            
            # Estimate noise floor using lower percentile
            noise_floor = np.percentile(envelope, 10)
            
            # Estimate signal using higher percentile
            signal_level = np.percentile(envelope, 90)
            
            # Calculate SNR
            if noise_floor > 0:
                snr = 20 * np.log10(signal_level / noise_floor)
            else:
                snr = 40  # Default high value if noise floor is extremely low
        
        except Exception as e:
            # Fallback method
            if signal.size == 0:
                return 0
                
            signal_power = np.mean(signal ** 2)
            # Estimate noise using signal variations
            noise_power = np.var(signal)
            
            if noise_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
            else:
                snr = 30  # Default value
                
        return float(snr)
    
    def perceptual_enhancement_filter(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Apply perceptual enhancement based on psychoacoustic principles.
        
        Args:
            audio (np.ndarray): Input audio signal.
            sr (int): Sample rate.
            
        Returns:
            np.ndarray: Enhanced audio.
        """
        try:
            # Convert to frequency domain
            n_fft = min(2048, len(audio))
            hop_length = min(512, n_fft // 4)
            
            # Compute STFT
            stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
            mag, phase = librosa.magphase(stft)
            
            # Calculate frequency bins
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Apply perceptual weighting based on equal-loudness contours
            # Simplified approximation of ISO226:2003 equal-loudness contours
            def iso_226_weight(f):
                # Approximation of 40-phon equal-loudness curve
                a = 4.47e-3
                f1 = 1000.0
                f2 = 1500.0
                f3 = 3000.0
                if f < 20:
                    return 0.0
                elif f < f1:
                    return 1.0 - np.exp(-a * f)
                elif f < f2:
                    return 1.0
                elif f < f3:
                    return 1.0 - 0.3 * (f - f2) / (f3 - f2)
                else:
                    return 0.7 - 0.3 * (f - f3) / 10000
            
            # Apply perceptual weighting to each frequency bin
            weights = np.array([iso_226_weight(f) for f in freqs])
            
            # Apply weights to magnitude
            enhanced_mag = mag * weights[:, np.newaxis]
            
            # Reconstruct signal
            enhanced_stft = enhanced_mag * phase
            enhanced_audio = librosa.istft(enhanced_stft, hop_length=hop_length, length=len(audio))
            
            return enhanced_audio
        except Exception as e:
            print(f"Error in perceptual enhancement: {str(e)}")
            return audio
    
    def harmonic_enhancement_filter(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Enhance harmonic components in the audio to improve speech clarity.
        
        Args:
            audio (np.ndarray): Input audio signal.
            sr (int): Sample rate.
            
        Returns:
            np.ndarray: Audio with enhanced harmonics.
        """
        try:
            # Skip for very short signals
            if len(audio) < sr // 4:
                return audio
                
            # Extract pitch and harmonics
            # Use librosa's harmonic-percussive source separation
            y_harmonic, y_percussive = librosa.effects.hpss(audio)
            
            # Enhance harmonic part
            # Speech is more prominent in harmonic components
            enhanced = audio * 0.7 + y_harmonic * 0.3
            
            return enhanced
        except Exception as e:
            print(f"Error in harmonic enhancement: {str(e)}")
            return audio
    
    def vocal_clarity_enhancement(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Enhance vocal clarity by focusing on speech frequency range.
        
        Args:
            audio (np.ndarray): Input audio signal.
            sr (int): Sample rate.
            
        Returns:
            np.ndarray: Audio with enhanced vocal clarity.
        """
        try:
            # Simple bandpass filter focusing on speech frequencies (300-3000 Hz)
            nyquist = sr // 2
            low_cutoff = 300 / nyquist
            high_cutoff = min(3000 / nyquist, 0.99)
            
            # Apply a bandpass filter to focus on speech frequencies
            b, a = signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            filtered = signal.filtfilt(b, a, audio)
            
            # Mix with original to preserve some natural characteristics
            enhanced = audio * 0.7 + filtered * 0.3
            
            return enhanced
        except Exception as e:
            print(f"Error in vocal clarity enhancement: {str(e)}")
            return audio
    
    def apply_dynamic_range_compression(self, audio: np.ndarray, threshold: float = -20.0, 
                                       ratio: float = 4.0, attack_ms: float = 5.0, 
                                       release_ms: float = 50.0) -> np.ndarray:
        """
        Apply dynamic range compression to audio signal.
        
        Args:
            audio (np.ndarray): Input audio signal.
            threshold (float): Threshold in dB.
            ratio (float): Compression ratio.
            attack_ms (float): Attack time in milliseconds.
            release_ms (float): Release time in milliseconds.
            
        Returns:
            np.ndarray: Compressed audio.
        """
        try:
            # Skip for very short signals
            if len(audio) < 1000:
                return audio
                
            # Convert to dB
            def to_db(x):
                return 20 * np.log10(np.maximum(np.abs(x), 1e-10))
                
            def from_db(x):
                return 10 ** (x / 20)
            
            # Calculate signal level
            db = to_db(audio)
            
            # Apply compression
            db_compressed = np.copy(db)
            mask = db > threshold
            db_compressed[mask] = threshold + (db[mask] - threshold) / ratio
            
            # Convert back to linear scale
            compressed = from_db(db_compressed) * np.sign(audio)
            
            # Normalize to match input level
            if np.max(np.abs(compressed)) > 0:
                compressed = compressed * (np.max(np.abs(audio)) / np.max(np.abs(compressed)))
            
            return compressed
        except Exception as e:
            print(f"Error in dynamic range compression: {str(e)}")
            return audio
    
    def denoise(self, audio_chunks: torch.Tensor, max_batch_size: Optional[int] = None) -> torch.Tensor:
        """
        Denoises noisy audio chunks with novel enhancements.

        Args:
            audio_chunks (torch.Tensor): Tensor of all noisy audio chunks.
            max_batch_size (int, optional): Same as for model initialization. Defaults to None.

        Returns:
            torch.Tensor: Denoised audio tensors.
        """
        if max_batch_size is None:
            max_batch_size = self.max_batch_size
        
        # Ensure audio_chunks has the right shape for the model
        if audio_chunks.dim() == 2:
            # Convert (num_chunks, chunk_length) to (num_chunks, 1, chunk_length)
            audio_chunks = audio_chunks.unsqueeze(1)
            
        num_chunks = audio_chunks.shape[0]
        batches = torch.split(audio_chunks, max_batch_size)
        
        if self.verbose:
            print("*" * 20)
            print(f"Total number of chunks: {num_chunks}")
            print(f"Processing {len(batches)} batches with up to {max_batch_size} chunks each")
            print("Audio chunks shape:", audio_chunks.shape)
            if self.adaptive_processing:
                print("Using adaptive SNR-based processing")
            if self.domain_adaptation:
                print(f"Using domain adaptation for: {self.domain_type}")
            print("*" * 20)
        
        denoised_audio = []
        for batch in tqdm(batches) if self.verbose else batches:
            # Prepare batch for model - should be (batch_size, chunk_length)
            # CleanUNet expects (batch_size, waveform) format
            if batch.dim() == 3:  # (batch_size, 1, chunk_length)
                batch_input = batch.squeeze(1)
            else:
                batch_input = batch
                
            # Move to device
            batch_input = batch_input.to(self.device)
            
            # Apply adaptive processing if enabled
            # Update the adaptive processing logic to be more conservative:
            if self.adaptive_processing:
                # Calculate SNR for each chunk in batch
                snr_values = []
                for chunk in batch_input:
                    # Move to CPU for SNR calculation
                    chunk_np = chunk.cpu().numpy()
                    snr = self.estimate_snr(chunk_np)
                    snr_values.append(snr)
                
                avg_snr = np.mean(snr_values)
                if self.verbose:
                    print(f"Batch average SNR: {avg_snr:.2f} dB")

                processing_mode = "standard"
                
                # More conservative thresholds and processing
                # Only apply very minimal adjustments to extremely clean sections
                if avg_snr > 30:  # Only for very clean audio
                    processing_mode = "light"
                    if self.verbose:
                        print(f"Batch with very high SNR ({avg_snr:.2f} dB): using minimal processing")

            
            # Process with model
            with torch.no_grad():
                batch_output = self.model(batch_input)
                
                # Apply domain adaptation if enabled
                if self.domain_adaptation:
                    batch_output = self.domain_adapter(batch_output)
            
            # Apply adaptive post-processing if needed
            # In the same denoise method:
            if self.adaptive_processing and processing_mode == "light":
                # Convert to numpy for processing
                batch_output_np = batch_output.cpu().numpy()
                
                # For light processing, blend a tiny bit of original to preserve details
                # but only for very clean segments
                for i in range(len(batch_output_np)):
                    orig_chunk = batch_input[i].cpu().numpy()
                    # Very subtle blend: 95% denoised, 5% original
                    batch_output_np[i] = 0.95 * batch_output_np[i] + 0.05 * orig_chunk
                
                # Convert back to tensor
                batch_output = torch.tensor(batch_output_np, device='cpu')
            
            # Prepare output for unchunk_audio
            batch_output = batch_output.unsqueeze(1)  # Add channel dimension
            
            # Move back to CPU and free GPU memory
            batch_output = batch_output.to('cpu')
            torch.cuda.empty_cache()
            
            # Unchunk this batch
            denoised_batch = unchunk_audio(batch_output)
            denoised_audio.append(denoised_batch)
            
        # Combine all batches
        denoised_audio = torch.cat(denoised_audio, dim=1)
        return denoised_audio
    
    def noise_classification(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """
        Classify the type of noise present in the audio.
        This is a simple feature extraction based approach.
        
        Args:
            audio (np.ndarray): Input audio signal.
            sr (int): Sample rate.
            
        Returns:
            Dict[str, float]: Dictionary of noise types and their probabilities.
        """
        try:
            # Extract audio features
            # 1. Spectral centroid - brightness
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            sc_mean = np.mean(spectral_centroid)
            
            # 2. Zero crossing rate - noisiness
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_mean = np.mean(zcr)
            
            # 3. Spectral bandwidth - spread
            bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
            bw_mean = np.mean(bandwidth)
            
            # 4. Spectral rolloff - high frequency content
            rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
            ro_mean = np.mean(rolloff)
            
            # Simplified noise classification based on features
            # These thresholds would normally be learned from data
            noise_types = {
                "white_noise": 0.0,
                "pink_noise": 0.0,
                "street_noise": 0.0,
                "crowd_noise": 0.0,
                "mechanical_noise": 0.0,
                "office_noise": 0.0,
            }
            
            # White noise: high ZCR, high centroid
            if zcr_mean > 0.1 and sc_mean > 3000:
                noise_types["white_noise"] = min(1.0, (zcr_mean - 0.1) * 5)
            
            # Pink noise: medium ZCR, dropping spectral energy
            if 0.05 < zcr_mean < 0.15 and 1500 < sc_mean < 3000:
                noise_types["pink_noise"] = min(1.0, max(0, (sc_mean - 1500) / 1500))
            
            # Street noise: medium ZCR, medium bandwidth, varying rolloff
            if 0.05 < zcr_mean < 0.2 and bw_mean > 2000:
                noise_types["street_noise"] = min(1.0, (bw_mean - 2000) / 1000)
            
            # Crowd noise: varying ZCR, medium-high bandwidth
            if 0.03 < zcr_mean < 0.15 and bw_mean > 1800:
                noise_types["crowd_noise"] = min(1.0, (bw_mean - 1800) / 1000)
            
            # Mechanical noise: low-medium ZCR, specific rolloff
            if zcr_mean < 0.1 and 1000 < ro_mean < 4000:
                noise_types["mechanical_noise"] = min(1.0, (ro_mean - 1000) / 3000)
            
            # Office noise: low ZCR, low centroid, moderate bandwidth
            if zcr_mean < 0.08 and sc_mean < 2000 and 500 < bw_mean < 2000:
                noise_types["office_noise"] = min(1.0, (2000 - sc_mean) / 1000)
            
            # Normalize probabilities
            total = sum(noise_types.values())
            if total > 0:
                for key in noise_types:
                    noise_types[key] /= total
            else:
                # If no clear classification, assign uniform probabilities
                for key in noise_types:
                    noise_types[key] = 1.0 / len(noise_types)
            
            return noise_types
        except Exception as e:
            print(f"Error in noise classification: {str(e)}")
            # Return uniform probabilities on error
            return {
                "white_noise": 0.167,
                "pink_noise": 0.167,
                "street_noise": 0.167,
                "crowd_noise": 0.167,
                "mechanical_noise": 0.167,
                "office_noise": 0.167,
            }
    
    def extract_audio_metrics(self, audio: np.ndarray, sr: int = 16000) -> Dict[str, float]:
        """
        Extract key audio metrics for analysis.
        
        Args:
            audio (np.ndarray): Input audio signal.
            sr (int): Sample rate.
            
        Returns:
            Dict[str, float]: Dictionary of audio metrics.
        """
        try:
            metrics = {}
            
            # Calculate RMS energy
            metrics["rms"] = float(np.sqrt(np.mean(audio**2)))
            
            # Calculate peak amplitude
            metrics["peak"] = float(np.max(np.abs(audio)))
            
            # Calculate dynamic range
            if metrics["peak"] > 0:
                metrics["dynamic_range_db"] = float(20 * np.log10(metrics["peak"] / (np.mean(np.abs(audio)) + 1e-10)))
            else:
                metrics["dynamic_range_db"] = 0.0
            
            # Extract spectral centroid (brightness)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
            metrics["spectral_centroid"] = float(np.mean(spectral_centroid))
            
            # Extract spectral contrast
            contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
            metrics["spectral_contrast"] = float(np.mean(contrast))
            
            # Calculate signal-to-noise ratio estimate
            metrics["estimated_snr"] = float(self.estimate_snr(audio))
            
            # Zero crossing rate (rough estimate of noisiness)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            metrics["zero_crossing_rate"] = float(np.mean(zcr))
            
            return metrics
        except Exception as e:
            print(f"Error calculating audio metrics: {str(e)}")
            return {
                "rms": 0.0,
                "peak": 0.0,
                "dynamic_range_db": 0.0,
                "spectral_centroid": 0.0,
                "spectral_contrast": 0.0,
                "estimated_snr": 0.0,
                "zero_crossing_rate": 0.0
            }
    
    def __call__(
        self, 
        noisy_audio_path: str,
        output_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        target_sr: int = 16000,
        clean_audio_path: Optional[str] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Denoises a given audio file with enhanced processing.

        Args:
            noisy_audio_path (str): Path to the noisy audio file.
            output_path (str, optional): Path to save the output. If None, don't save.
            metrics_path (str, optional): Path to save metrics and visualizations.
            target_sr (int, optional): Target sampling rate. Defaults to 16000.
            clean_audio_path (str, optional): Path to clean reference audio for metrics.

        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: 
                - Denoised audio waveform
                - Dictionary with metrics and analysis results
        """
        if self.verbose:
            print(f"Processing audio file: {noisy_audio_path}")
            print(f"Applied enhancements: Adaptive={self.adaptive_processing}, "
                  f"Harmonic={self.harmonic_enhancement}, "
                  f"Vocal={self.vocal_clarity}, "
                  f"DRC={self.dynamic_range_compression}, "
                  f"Domain-adaptation={self.domain_adaptation}")
        
        # Load and normalize audio
        noisy_audio = self.load_audio_and_resample(audio_path=noisy_audio_path, target_sr=target_sr)
        noisy_audio_np = self.audio_processing(audio_wav=noisy_audio)
        
        # Store original noisy audio for metrics
        original_noisy = np.copy(noisy_audio_np)
        
        # Extract metrics from original audio
        original_metrics = self.extract_audio_metrics(original_noisy, sr=target_sr)
        
        # Classify noise type
        noise_classification = self.noise_classification(original_noisy, sr=target_sr)
        
        if self.verbose:
            print(f"Loaded audio with shape: {noisy_audio_np.shape}")
            print(f"Detected noise types: {', '.join([f'{k}: {v:.2f}' for k, v in noise_classification.items() if v > 0.1])}")
        
        # Create chunks
        audio_chunks = chunk_audio(
            audio_signal=noisy_audio_np, 
            sampling_rate=target_sr, 
            chunk_length_sec=self.chunk_length_s
        )
        
        if self.verbose:
            print(f"Created {audio_chunks.shape[0]} chunks with shape: {audio_chunks.shape}")
        
        # Denoise chunks with enhanced processing
        denoised_audio = self.denoise(audio_chunks=audio_chunks)
        
        if self.verbose:
            print(f"Denoised audio shape: {denoised_audio.shape}")
        
        # Process output
        denoised_audio_np = self.audio_processing(denoised_audio[0])
        
        # Apply perceptual enhancement (always active)
        if self._perceptual_enhancement:
            if self.verbose:
                print("Applying perceptual enhancement...")
            denoised_audio_np = self.perceptual_enhancement_filter(denoised_audio_np, sr=target_sr)
        
        # Apply harmonic enhancement if enabled
        if self.harmonic_enhancement:
            if self.verbose:
                print("Applying harmonic enhancement...")
            denoised_audio_np = self.harmonic_enhancement_filter(denoised_audio_np, sr=target_sr)
        
        # Apply vocal clarity enhancement if enabled
        if self.vocal_clarity:
            if self.verbose:
                print("Applying vocal clarity enhancement...")
            denoised_audio_np = self.vocal_clarity_enhancement(denoised_audio_np, sr=target_sr)
        
        # Apply dynamic range compression if enabled
        if self.dynamic_range_compression:
            if self.verbose:
                print("Applying dynamic range compression...")
            denoised_audio_np = self.apply_dynamic_range_compression(denoised_audio_np)
        
        # Extract metrics from processed audio
        processed_metrics = self.extract_audio_metrics(denoised_audio_np, sr=target_sr)
        
        # Calculate improvement metrics
        improvement_metrics = {
            "snr_improvement": processed_metrics["estimated_snr"] - original_metrics["estimated_snr"],
            "peak_reduction": original_metrics["peak"] - processed_metrics["peak"],
            "spectral_balance_change": processed_metrics["spectral_centroid"] - original_metrics["spectral_centroid"],
            "noise_reduction": original_metrics["zero_crossing_rate"] - processed_metrics["zero_crossing_rate"]
        }
        
        # Compile all metrics and analysis results
        result_metrics = {
            "original": original_metrics,
            "processed": processed_metrics,
            "improvement": improvement_metrics,
            "noise_classification": noise_classification,
            "processing_info": {
                "adaptive_processing": self.adaptive_processing,
                "harmonic_enhancement": self.harmonic_enhancement,
                "vocal_clarity": self.vocal_clarity,
                "dynamic_range_compression": self.dynamic_range_compression,
                "perceptual_enhancement": self._perceptual_enhancement
            }
        }
        
        # Generate metrics if clean reference is provided
        if clean_audio_path and metrics_path:
            try:
                from visualization_utils import AudioMetrics
                
                # Load clean reference audio
                clean_audio = self.load_audio_and_resample(audio_path=clean_audio_path, target_sr=target_sr)
                clean_audio_np = self.audio_processing(audio_wav=clean_audio)
                
                # Create metrics report
                if self.verbose:
                    print(f"Generating metrics report at: {metrics_path}")
                    
                report_path = AudioMetrics.create_full_report(
                    clean_audio=clean_audio_np,
                    noisy_audio=original_noisy,
                    denoised_audio=denoised_audio_np,
                    sr=target_sr,
                    output_dir=metrics_path
                )
                
                # Add reference metrics to results
                result_metrics["reference_metrics"] = report_path
                
            except Exception as e:
                print(f"Error generating metrics: {str(e)}")
        
        # Save if output path is provided
        if output_path:
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            # Ensure the audio is normalized to prevent clipping
            if np.max(np.abs(denoised_audio_np)) > 0:
                denoised_audio_np = denoised_audio_np / np.max(np.abs(denoised_audio_np)) * 0.9
            
            try:
                # Use soundfile for better compatibility with PCM formats
                import soundfile as sf
                # Save as 16-bit PCM WAV - widely compatible format
                sf.write(output_path, denoised_audio_np, target_sr, subtype='PCM_16')
                
                if self.verbose:
                    print(f"Saved denoised audio to: {output_path} (16-bit PCM format)")
            except Exception as e:
                print(f"Error with soundfile: {e}")
                # Fall back to torchaudio if soundfile fails
                denoised_tensor = torch.from_numpy(denoised_audio_np).unsqueeze(0)
                # Convert to 16-bit int format
                denoised_tensor = (denoised_tensor * 32767).to(torch.int16)
                torchaudio.save(output_path, denoised_tensor, target_sr)
                if self.verbose:
                    print(f"Saved denoised audio to: {output_path} (using torchaudio)")
        
        return denoised_audio_np, result_metrics


class DomainAdapter(torch.nn.Module):
    """
    Lightweight adaptation layer for domain-specific denoising.
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        # Small 1D convolutional network for adaptation
        self.conv1 = torch.nn.Conv1d(1, hidden_dim, kernel_size=15, padding=7)
        self.conv2 = torch.nn.Conv1d(hidden_dim, hidden_dim, kernel_size=15, padding=7)
        self.conv3 = torch.nn.Conv1d(hidden_dim, 1, kernel_size=15, padding=7)
        self.relu = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(hidden_dim)
        self.bn2 = torch.nn.BatchNorm1d(hidden_dim)
        
    def forward(self, x):
        # Input shape: (batch_size, chunk_length)
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 1, chunk_length)
        
        # First conv layer with batch norm and activation
        x = self.relu(self.bn1(self.conv1(x)))
        
        # Second conv layer with batch norm and activation
        x = self.relu(self.bn2(self.conv2(x)))
        
        # Output layer
        x = self.conv3(x)
        
        # Remove channel dimension
        return x.squeeze(1)