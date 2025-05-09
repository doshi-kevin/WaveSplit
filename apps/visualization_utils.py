import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import torch
import torchaudio
import os
from pesq import pesq
from pystoi import stoi

class AudioMetrics:
    """
    Class for computing and visualizing audio quality metrics and spectrograms
    """
    
    @staticmethod
    def compute_metrics(clean_audio, noisy_audio, denoised_audio, sr=16000):
        """
        Compute various audio quality metrics
        
        Args:
            clean_audio: np.ndarray - Clean reference audio
            noisy_audio: np.ndarray - Noisy input audio
            denoised_audio: np.ndarray - Denoised output audio
            sr: int - Sample rate
            
        Returns:
            dict: Dictionary containing all metrics
        """
        # Ensure all signals have the same length (use shortest length)
        min_len = min(len(clean_audio), len(noisy_audio), len(denoised_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        denoised_audio = denoised_audio[:min_len]
        
        # Calculate SNR
        def calculate_snr(clean, processed):
            noise = clean - processed
            return 10 * np.log10(np.sum(clean**2) / np.sum(noise**2))
        
        # Calculate PESQ (Perceptual Evaluation of Speech Quality)
        try:
            noisy_pesq = pesq(sr, clean_audio, noisy_audio, 'wb')
            denoised_pesq = pesq(sr, clean_audio, denoised_audio, 'wb')
        except Exception as e:
            print(f"Error calculating PESQ: {e}")
            noisy_pesq = float('nan')
            denoised_pesq = float('nan')
            
        # Calculate STOI (Short-Time Objective Intelligibility)
        try:
            noisy_stoi = stoi(clean_audio, noisy_audio, sr, extended=False)
            denoised_stoi = stoi(clean_audio, denoised_audio, sr, extended=False)
        except Exception as e:
            print(f"Error calculating STOI: {e}")
            noisy_stoi = float('nan')
            denoised_stoi = float('nan')
            
        # Calculate SNR
        noisy_snr = calculate_snr(clean_audio, noisy_audio)
        denoised_snr = calculate_snr(clean_audio, denoised_audio)
        
        # Compute improvement metrics
        snr_improvement = denoised_snr - noisy_snr
        pesq_improvement = denoised_pesq - noisy_pesq
        stoi_improvement = denoised_stoi - noisy_stoi
        
        return {
            "noisy_snr": noisy_snr,
            "denoised_snr": denoised_snr,
            "snr_improvement": snr_improvement,
            "noisy_pesq": noisy_pesq,
            "denoised_pesq": denoised_pesq,
            "pesq_improvement": pesq_improvement,
            "noisy_stoi": noisy_stoi,
            "denoised_stoi": denoised_stoi,
            "stoi_improvement": stoi_improvement
        }
    
    # Add this function to apps/visualization_utils.py file:

    @staticmethod
    def create_comparison_plots(noisy_audio, denoised_audio, sr=16000, output_dir=None):
        """
        Create comprehensive comparison plots between noisy and denoised audio
        
        Args:
            noisy_audio: np.ndarray - Noisy input audio
            denoised_audio: np.ndarray - Denoised output audio
            sr: int - Sample rate
            output_dir: str - Directory to save plots
            
        Returns:
            dict: Dictionary of plot file paths
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import librosa
        import librosa.display
        import scipy.signal
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        # Ensure both audio arrays have the same length
        # This is critical to prevent shape mismatch errors
        min_length = min(len(noisy_audio), len(denoised_audio))
        noisy_audio = noisy_audio[:min_length]
        denoised_audio = denoised_audio[:min_length]
        
        print(f"Debug - Audio shapes after trimming: noisy={noisy_audio.shape}, denoised={denoised_audio.shape}")
        
        # 1. Waveform comparison
        plt.figure(figsize=(12, 6))
        time = np.arange(0, len(noisy_audio)) / sr
        
        plt.subplot(2, 1, 1)
        plt.plot(time, noisy_audio)
        plt.title('Noisy Audio Waveform')
        plt.ylabel('Amplitude')
        plt.grid(alpha=0.3)
        
        plt.subplot(2, 1, 2)
        plt.plot(time, denoised_audio)
        plt.title('Denoised Audio Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        waveform_path = os.path.join(output_dir, 'waveform_comparison.png') if output_dir else None
        if waveform_path:
            plt.savefig(waveform_path, dpi=300, bbox_inches='tight')
            results['waveform'] = waveform_path
        plt.close()
        
        try:
            # 2. Power spectral density comparison
            plt.figure(figsize=(12, 6))
            
            # Compute PSD - use the same number of segments for both
            nperseg = min(2048, min_length // 8)  # Ensure nperseg is small enough for the audio length
            freqs_noisy, psd_noisy = scipy.signal.welch(noisy_audio, sr, nperseg=nperseg)
            freqs_denoised, psd_denoised = scipy.signal.welch(denoised_audio, sr, nperseg=nperseg)
            
            # Ensure the shapes match
            min_freq_len = min(len(freqs_noisy), len(freqs_denoised))
            freqs_noisy = freqs_noisy[:min_freq_len]
            psd_noisy = psd_noisy[:min_freq_len]
            freqs_denoised = freqs_denoised[:min_freq_len]
            psd_denoised = psd_denoised[:min_freq_len]
            
            # Convert to dB and handle potential zeros
            psd_noisy_db = 10 * np.log10(np.maximum(psd_noisy, 1e-10))
            psd_denoised_db = 10 * np.log10(np.maximum(psd_denoised, 1e-10))
            
            plt.semilogx(freqs_noisy, psd_noisy_db, label='Noisy')
            plt.semilogx(freqs_denoised, psd_denoised_db, label='Denoised')
            plt.title('Power Spectral Density Comparison')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power/Frequency (dB/Hz)')
            plt.grid(True, which="both", ls="-", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            
            psd_path = os.path.join(output_dir, 'psd_comparison.png') if output_dir else None
            if psd_path:
                plt.savefig(psd_path, dpi=300, bbox_inches='tight')
                results['psd'] = psd_path
            plt.close()
        except Exception as e:
            print(f"Error generating PSD plot: {e}")
        
        try:
            # 3. Spectrogram difference visualization (highlights removed noise)
            plt.figure(figsize=(12, 6))
            
            # Use the same STFT parameters for both
            n_fft = min(2048, min_length // 2)
            hop_length = n_fft // 4
            
            # Compute spectrograms
            S_noisy = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)
            S_denoised = librosa.stft(denoised_audio, n_fft=n_fft, hop_length=hop_length)
            
            # Convert to dB scale
            S_noisy_db = librosa.amplitude_to_db(np.abs(S_noisy), ref=np.max)
            S_denoised_db = librosa.amplitude_to_db(np.abs(S_denoised), ref=np.max)
            
            # Calculate the difference (what was removed by denoising)
            # Ensure shapes match first
            min_time_frames = min(S_noisy_db.shape[1], S_denoised_db.shape[1])
            diff = S_noisy_db[:, :min_time_frames] - S_denoised_db[:, :min_time_frames]
            
            # Plot the difference spectrogram
            img = librosa.display.specshow(
                diff, y_axis='log', x_axis='time', sr=sr, hop_length=hop_length,
                cmap='magma', vmin=0, vmax=20  # Only show positive differences (removed noise)
            )
            plt.colorbar(img, format='%+2.0f dB', ax=plt.gca())
            plt.title('Noise Reduction Map (Removed Components)')
            plt.tight_layout()
            
            diff_path = os.path.join(output_dir, 'noise_reduction_map.png') if output_dir else None
            if diff_path:
                plt.savefig(diff_path, dpi=300, bbox_inches='tight')
                results['diff'] = diff_path
            plt.close()
        except Exception as e:
            print(f"Error generating difference spectrogram: {e}")
        
        try:
            # 4. Energy distribution comparison (frequency bands)
            plt.figure(figsize=(12, 6))
            
            # Calculate frequency bands energies
            n_fft = min(2048, min_length // 2)
            
            # Compute STFT
            X_noisy = librosa.stft(noisy_audio, n_fft=n_fft)
            X_denoised = librosa.stft(denoised_audio, n_fft=n_fft)
            
            # Compute power
            noisy_power = np.abs(X_noisy)**2
            denoised_power = np.abs(X_denoised)**2
            
            # Get frequency bins
            freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
            
            # Group into octave bands for visualization
            n_bands = 8
            # Ensure reasonable frequency range
            band_edges = np.logspace(np.log10(50), np.log10(min(sr/2, 8000)), n_bands+1)
            
            noisy_band_power = []
            denoised_band_power = []
            band_labels = []
            
            for i in range(n_bands):
                lo, hi = band_edges[i], band_edges[i+1]
                band_mask = (freqs >= lo) & (freqs < hi)
                
                if np.any(band_mask):  # Check if any frequencies are in this band
                    noisy_band_power.append(np.mean(noisy_power[band_mask, :]))
                    denoised_band_power.append(np.mean(denoised_power[band_mask, :]))
                    band_labels.append(f"{int(lo)}-{int(hi)} Hz")
            
            # Convert to dB with safeguards
            max_noisy_power = max(np.max(noisy_band_power), 1e-10)
            noisy_band_db = librosa.amplitude_to_db(np.maximum(noisy_band_power, 1e-10), ref=max_noisy_power)
            denoised_band_db = librosa.amplitude_to_db(np.maximum(denoised_band_power, 1e-10), ref=max_noisy_power)
            
            # Plot
            x = np.arange(len(band_labels))
            width = 0.35
            
            plt.bar(x - width/2, noisy_band_db, width, label='Noisy')
            plt.bar(x + width/2, denoised_band_db, width, label='Denoised')
            
            plt.xlabel('Frequency Band')
            plt.ylabel('Energy (dB)')
            plt.title('Frequency Band Energy Distribution')
            plt.xticks(x, band_labels, rotation=45)
            plt.legend()
            plt.tight_layout()
            
            energy_path = os.path.join(output_dir, 'energy_distribution.png') if output_dir else None
            if energy_path:
                plt.savefig(energy_path, dpi=300, bbox_inches='tight')
                results['energy'] = energy_path
            plt.close()
        except Exception as e:
            print(f"Error generating energy distribution plot: {e}")
        
        return results

    def compute_advanced_metrics(clean_audio, noisy_audio, denoised_audio, sr=16000):
        """
        Compute comprehensive audio quality metrics including PESQ and STOI
        
        Args:
            clean_audio: np.ndarray - Clean reference audio
            noisy_audio: np.ndarray - Noisy input audio
            denoised_audio: np.ndarray - Denoised output audio
            sr: int - Sample rate
            
        Returns:
            dict: Dictionary containing all metrics
        """
        # Make sure all arrays have the same length
        min_len = min(len(clean_audio), len(noisy_audio), len(denoised_audio))
        clean_audio = clean_audio[:min_len]
        noisy_audio = noisy_audio[:min_len]
        denoised_audio = denoised_audio[:min_len]
        
        # Calculate SNR
        def calculate_snr(clean, processed):
            noise = clean - processed
            return 10 * np.log10(np.sum(clean**2) / np.sum(noise**2))
        
        # Calculate PESQ (Perceptual Evaluation of Speech Quality)
        try:
            noisy_pesq = pesq(sr, clean_audio, noisy_audio, 'wb')
            denoised_pesq = pesq(sr, clean_audio, denoised_audio, 'wb')
        except Exception as e:
            print(f"Error calculating PESQ: {e}")
            noisy_pesq = float('nan')
            denoised_pesq = float('nan')
            
        # Calculate STOI (Short-Time Objective Intelligibility)
        try:
            noisy_stoi = stoi(clean_audio, noisy_audio, sr, extended=False)
            denoised_stoi = stoi(clean_audio, denoised_audio, sr, extended=False)
        except Exception as e:
            print(f"Error calculating STOI: {e}")
            noisy_stoi = float('nan')
            denoised_stoi = float('nan')
        
        # Calculate SDI (Speech Distortion Index) - Approximation
        def calculate_sdi(clean, processed):
            # A simple approximation of speech distortion
            # Lower values indicate less distortion
            error = clean - processed
            return np.sum(error**2) / np.sum(clean**2)
        
        noisy_sdi = calculate_sdi(clean_audio, noisy_audio)
        denoised_sdi = calculate_sdi(clean_audio, denoised_audio)
        
        # Calculate SNR
        noisy_snr = calculate_snr(clean_audio, noisy_audio)
        denoised_snr = calculate_snr(clean_audio, denoised_audio)
        
        # Calculate Frequency-Weighted Segmental SNR (fwSNRseg)
        def calculate_fwsnrseg(clean, processed, sr):
            from scipy import signal
            
            # Parameters
            window_size = int(sr * 0.03)  # 30ms window
            step = int(window_size / 2)   # 50% overlap
            
            # Frequency weightings (simplified version)
            weights = np.linspace(0.5, 4.0, 25)  # Emphasize mid frequencies
            
            total_fwsnr = 0
            num_segments = 0
            
            for i in range(0, len(clean) - window_size, step):
                # Get segment
                clean_seg = clean[i:i+window_size]
                processed_seg = processed[i:i+window_size]
                
                # Apply window
                win = signal.windows.hann(window_size)
                clean_seg = clean_seg * win
                processed_seg = processed_seg * win
                
                # Calculate spectra
                clean_spec = np.abs(np.fft.rfft(clean_seg))
                processed_spec = np.abs(np.fft.rfft(processed_seg))
                
                # Calculate SNR in each frequency band
                noise_spec = clean_spec - processed_spec
                eps = 1e-10  # To avoid division by zero
                
                # Trim to match weights length
                band_length = min(len(clean_spec), len(weights))
                
                # Calculate weighted SNR
                snr_bands = []
                for j in range(band_length):
                    if noise_spec[j] > eps:
                        snr_band = 10 * np.log10((clean_spec[j]**2) / (noise_spec[j]**2 + eps))
                        snr_band = max(-10, min(35, snr_band))  # Clamp values
                        snr_bands.append(snr_band * weights[j])
                    else:
                        snr_bands.append(35 * weights[j])  # Maximum value if noise is negligible
                
                if snr_bands:
                    fwsnr_seg = np.sum(snr_bands) / np.sum(weights[:len(snr_bands)])
                    total_fwsnr += fwsnr_seg
                    num_segments += 1
            
            if num_segments > 0:
                return total_fwsnr / num_segments
            else:
                return 0
        
        try:
            noisy_fwsnrseg = calculate_fwsnrseg(clean_audio, noisy_audio, sr)
            denoised_fwsnrseg = calculate_fwsnrseg(clean_audio, denoised_audio, sr)
        except Exception as e:
            print(f"Error calculating fwSNRseg: {e}")
            noisy_fwsnrseg = float('nan')
            denoised_fwsnrseg = float('nan')
        
        # Compile all metrics
        return {
            "noisy_snr": noisy_snr,
            "denoised_snr": denoised_snr,
            "snr_improvement": denoised_snr - noisy_snr,
            "noisy_pesq": noisy_pesq,
            "denoised_pesq": denoised_pesq,
            "pesq_improvement": denoised_pesq - noisy_pesq,
            "noisy_stoi": noisy_stoi,
            "denoised_stoi": denoised_stoi,
            "stoi_improvement": denoised_stoi - noisy_stoi,
            "noisy_sdi": noisy_sdi,
            "denoised_sdi": denoised_sdi,
            "sdi_improvement": noisy_sdi - denoised_sdi,  # Lower SDI is better
            "noisy_fwsnrseg": noisy_fwsnrseg,
            "denoised_fwsnrseg": denoised_fwsnrseg,
            "fwsnrseg_improvement": denoised_fwsnrseg - noisy_fwsnrseg
        }
        
        # Calculate SDI (Speech Distortion Index) - Approximation
        def calculate_sdi(clean, processed):
            # A simple approximation of speech distortion
            # Lower values indicate less distortion
            error = clean - processed
            return np.sum(error**2) / np.sum(clean**2)
        
        noisy_sdi = calculate_sdi(clean_audio, noisy_audio)
        denoised_sdi = calculate_sdi(clean_audio, denoised_audio)
        
        # Calculate SNR
        noisy_snr = calculate_snr(clean_audio, noisy_audio)
        denoised_snr = calculate_snr(clean_audio, denoised_audio)
        
        # Calculate Frequency-Weighted Segmental SNR (fwSNRseg)
        def calculate_fwsnrseg(clean, processed, sr):
            from scipy import signal
            
            # Parameters
            window_size = int(sr * 0.03)  # 30ms window
            step = int(window_size / 2)   # 50% overlap
            
            # Frequency weightings (simplified version)
            weights = np.linspace(0.5, 4.0, 25)  # Emphasize mid frequencies
            
            total_fwsnr = 0
            num_segments = 0
            
            for i in range(0, len(clean) - window_size, step):
                # Get segment
                clean_seg = clean[i:i+window_size]
                processed_seg = processed[i:i+window_size]
                
                # Apply window
                win = signal.windows.hann(window_size)
                clean_seg = clean_seg * win
                processed_seg = processed_seg * win
                
                # Calculate spectra
                clean_spec = np.abs(np.fft.rfft(clean_seg))
                processed_spec = np.abs(np.fft.rfft(processed_seg))
                
                # Calculate SNR in each frequency band
                noise_spec = clean_spec - processed_spec
                eps = 1e-10  # To avoid division by zero
                
                # Trim to match weights length
                band_length = min(len(clean_spec), len(weights))
                
                # Calculate weighted SNR
                snr_bands = []
                for j in range(band_length):
                    if noise_spec[j] > eps:
                        snr_band = 10 * np.log10((clean_spec[j]**2) / (noise_spec[j]**2 + eps))
                        snr_band = max(-10, min(35, snr_band))  # Clamp values
                        snr_bands.append(snr_band * weights[j])
                    else:
                        snr_bands.append(35 * weights[j])  # Maximum value if noise is negligible
                
                if snr_bands:
                    fwsnr_seg = np.sum(snr_bands) / np.sum(weights[:len(snr_bands)])
                    total_fwsnr += fwsnr_seg
                    num_segments += 1
            
            if num_segments > 0:
                return total_fwsnr / num_segments
            else:
                return 0
        
        try:
            noisy_fwsnrseg = calculate_fwsnrseg(clean_audio, noisy_audio, sr)
            denoised_fwsnrseg = calculate_fwsnrseg(clean_audio, denoised_audio, sr)
        except Exception as e:
            print(f"Error calculating fwSNRseg: {e}")
            noisy_fwsnrseg = float('nan')
            denoised_fwsnrseg = float('nan')
        
        # Compile all metrics
        return {
            "noisy_snr": noisy_snr,
            "denoised_snr": denoised_snr,
            "snr_improvement": denoised_snr - noisy_snr,
            "noisy_pesq": noisy_pesq,
            "denoised_pesq": denoised_pesq,
            "pesq_improvement": denoised_pesq - noisy_pesq,
            "noisy_stoi": noisy_stoi,
            "denoised_stoi": denoised_stoi,
            "stoi_improvement": denoised_stoi - noisy_stoi,
            "noisy_sdi": noisy_sdi,
            "denoised_sdi": denoised_sdi,
            "sdi_improvement": noisy_sdi - denoised_sdi,  # Lower SDI is better
            "noisy_fwsnrseg": noisy_fwsnrseg,
            "denoised_fwsnrseg": denoised_fwsnrseg,
            "fwsnrseg_improvement": denoised_fwsnrseg - noisy_fwsnrseg
        }
                
        @staticmethod
        def plot_waveforms(clean_audio=None, noisy_audio=None, denoised_audio=None, 
                            sr=16000, output_path=None, fig_size=(15, 9)):
            """
            Plot waveforms for comparison
            
            Args:
                clean_audio: np.ndarray or None - Clean reference audio
                noisy_audio: np.ndarray or None - Noisy input audio
                denoised_audio: np.ndarray or None - Denoised output audio
                sr: int - Sample rate
                output_path: str - Path to save the figure
                fig_size: tuple - Figure size
                
            Returns:
                plt.Figure: Figure object or saves to disk if output path provided
            """
            # Count valid inputs
            plot_count = sum(x is not None for x in [clean_audio, noisy_audio, denoised_audio])
            
            if plot_count == 0:
                raise ValueError("At least one audio input must be provided")
                
            plt.figure(figsize=fig_size)
            
            plot_idx = 1
            titles = ["Clean Audio", "Noisy Audio", "Denoised Audio"]
            audio_types = [clean_audio, noisy_audio, denoised_audio]
            
            # Ensure all have the same length for time alignment
            valid_audios = [x for x in audio_types if x is not None]
            min_length = min(len(x) for x in valid_audios)
            
            for i, (title, audio) in enumerate(zip(titles, audio_types)):
                if audio is None:
                    continue
                    
                plt.subplot(plot_count, 1, plot_idx)
                
                # Trim to common length
                audio_trimmed = audio[:min_length]
                
                # Time axis
                time = np.arange(0, len(audio_trimmed)) / sr
                
                plt.plot(time, audio_trimmed)
                plt.title(title)
                plt.xlabel("Time (s)")
                plt.ylabel("Amplitude")
                plt.tight_layout()
                plot_idx += 1
                
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                return output_path
            else:
                return plt.gcf()
            
    @staticmethod
    def plot_spectrograms(clean_audio=None, noisy_audio=None, denoised_audio=None, 
                        sr=16000, output_path=None, fig_size=(18, 12)):
        """
        Generate and plot spectrograms for comparison
        
        Args:
            clean_audio: np.ndarray or None - Clean reference audio
            noisy_audio: np.ndarray or None - Noisy input audio
            denoised_audio: np.ndarray or None - Denoised output audio
            sr: int - Sample rate
            output_path: str - Path to save the figure
            fig_size: tuple - Figure size
            
        Returns:
            plt.Figure: Figure object or saves to disk if output path provided
        """
        # Count how many spectrograms to plot
        plot_count = sum(x is not None for x in [clean_audio, noisy_audio, denoised_audio])
        
        if plot_count == 0:
            raise ValueError("At least one audio input must be provided")
            
        plt.figure(figsize=fig_size)
        
        plot_idx = 1
        
        # Prepare for consistent color scaling
        max_db = float('-inf')
        min_db = float('inf')
        
        # First pass to find common dB range
        for audio in [x for x in [clean_audio, noisy_audio, denoised_audio] if x is not None]:
            S = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            max_db = max(max_db, np.max(S))
            min_db = min(min_db, np.min(S))
            
        # Plot with consistent color scaling
        titles = ["Clean Audio", "Noisy Audio", "Denoised Audio"]
        audio_types = [clean_audio, noisy_audio, denoised_audio]
        
        for i, (title, audio) in enumerate(zip(titles, audio_types)):
            if audio is None:
                continue
                
            plt.subplot(plot_count, 1, plot_idx)
            
            # Compute spectrogram
            S = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            
            # Plot spectrogram with consistent color range
            img = librosa.display.specshow(
                S, y_axis='log', x_axis='time', sr=sr, cmap='viridis',
                vmin=min_db, vmax=max_db
            )
            
            plt.title(f"{title} Spectrogram")
            plt.tight_layout()
            plot_idx += 1
            
        plt.colorbar(img, format='%+2.0f dB', ax=plt.gcf().axes)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            return plt.gcf()
    
    @staticmethod
    def create_full_report(clean_audio=None, noisy_audio=None, denoised_audio=None, 
                           sr=16000, output_dir='metrics_report'):
        """
        Create a full report with metrics and visualizations
        
        Args:
            clean_audio: np.ndarray or None - Clean reference audio
            noisy_audio: np.ndarray or None - Noisy input audio
            denoised_audio: np.ndarray or None - Denoised output audio
            sr: int - Sample rate
            output_dir: str - Directory to save the report
            
        Returns:
            str: Path to the report directory
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate metrics if clean audio is available
        metrics = None
        if clean_audio is not None and noisy_audio is not None and denoised_audio is not None:
            metrics = AudioMetrics.compute_metrics(
                clean_audio, noisy_audio, denoised_audio, sr
            )
            
            # Save metrics to file
            with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
                f.write("=== Audio Quality Metrics ===\n\n")
                f.write(f"SNR (Noisy): {metrics['noisy_snr']:.2f} dB\n")
                f.write(f"SNR (Denoised): {metrics['denoised_snr']:.2f} dB\n")
                f.write(f"SNR Improvement: {metrics['snr_improvement']:.2f} dB\n\n")
                
                f.write(f"PESQ (Noisy): {metrics['noisy_pesq']:.3f}\n")
                f.write(f"PESQ (Denoised): {metrics['denoised_pesq']:.3f}\n")
                f.write(f"PESQ Improvement: {metrics['pesq_improvement']:.3f}\n\n")
                
                f.write(f"STOI (Noisy): {metrics['noisy_stoi']:.3f}\n")
                f.write(f"STOI (Denoised): {metrics['denoised_stoi']:.3f}\n")
                f.write(f"STOI Improvement: {metrics['stoi_improvement']:.3f}\n")
        
        # Generate spectrograms
        spectrogram_path = os.path.join(output_dir, 'spectrograms.png')
        AudioMetrics.plot_spectrograms(
            clean_audio, noisy_audio, denoised_audio, sr, spectrogram_path
        )
        
        # Generate waveform plots
        waveform_path = os.path.join(output_dir, 'waveforms.png')
        AudioMetrics.plot_waveforms(
            clean_audio, noisy_audio, denoised_audio, sr, waveform_path
        )
        
        return output_dir
    

    @staticmethod
    def plot_enhanced_comparisons(noisy_audio, denoised_audio, sr=16000, output_path=None, fig_size=(18, 15)):
        """
        Generate enhanced comparison visualizations between noisy and denoised audio
        
        Args:
            noisy_audio: np.ndarray - Noisy input audio
            denoised_audio: np.ndarray - Denoised output audio
            sr: int - Sample rate
            output_path: str - Path to save the figure
            fig_size: tuple - Figure size
            
        Returns:
            plt.Figure: Figure object or saves to disk if output path provided
        """
        if noisy_audio is None or denoised_audio is None:
            raise ValueError("Both noisy and denoised audio must be provided")
            
        # Ensure same length
        min_length = min(len(noisy_audio), len(denoised_audio))
        noisy_audio = noisy_audio[:min_length]
        denoised_audio = denoised_audio[:min_length]
        
        plt.figure(figsize=fig_size)
        
        # 1. Waveform comparison (overlay)
        plt.subplot(3, 1, 1)
        time = np.arange(0, len(noisy_audio)) / sr
        plt.plot(time, noisy_audio, alpha=0.7, label='Noisy')
        plt.plot(time, denoised_audio, alpha=0.7, label='Denoised')
        plt.title('Waveform Comparison', fontsize=14)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # 2. Spectrogram difference
        plt.subplot(3, 1, 2)
        S_noisy = librosa.amplitude_to_db(np.abs(librosa.stft(noisy_audio)), ref=np.max)
        S_denoised = librosa.amplitude_to_db(np.abs(librosa.stft(denoised_audio)), ref=np.max)
        
        # Calculate the difference
        diff = S_denoised - S_noisy
        
        # Plot the difference spectrogram
        img = librosa.display.specshow(
            diff, y_axis='log', x_axis='time', sr=sr, cmap='coolwarm',
            vmin=-10, vmax=10  # Symmetric colormap range
        )
        plt.colorbar(img, format='%+2.0f dB', ax=plt.gca())
        plt.title('Spectrogram Difference (Denoised - Noisy)', fontsize=14)
        
        # 3. Energy distribution comparison
        plt.subplot(3, 1, 3)
        
        # Calculate frequency bands energies
        n_fft = 2048
        
        # Compute STFT
        X_noisy = librosa.stft(noisy_audio, n_fft=n_fft)
        X_denoised = librosa.stft(denoised_audio, n_fft=n_fft)
        
        # Compute power
        noisy_power = np.abs(X_noisy)**2
        denoised_power = np.abs(X_denoised)**2
        
        # Get frequency bins
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        # Group into octave bands for visualization
        n_bands = 8
        band_edges = np.logspace(np.log10(50), np.log10(sr/2), n_bands+1)
        
        noisy_band_power = []
        denoised_band_power = []
        band_labels = []
        
        for i in range(n_bands):
            lo, hi = band_edges[i], band_edges[i+1]
            band_mask = (freqs >= lo) & (freqs < hi)
            
            noisy_band_power.append(np.mean(noisy_power[band_mask, :]))
            denoised_band_power.append(np.mean(denoised_power[band_mask, :]))
            band_labels.append(f"{int(lo)}-{int(hi)} Hz")
        
        # Convert to dB
        noisy_band_db = librosa.amplitude_to_db(noisy_band_power, ref=np.max(noisy_band_power))
        denoised_band_db = librosa.amplitude_to_db(denoised_band_power, ref=np.max(noisy_band_power))  # Use same ref
        
        # Plot
        x = np.arange(len(band_labels))
        width = 0.35
        
        plt.bar(x - width/2, noisy_band_db, width, label='Noisy')
        plt.bar(x + width/2, denoised_band_db, width, label='Denoised')
        
        plt.xlabel('Frequency Band')
        plt.ylabel('Energy (dB)')
        plt.title('Frequency Band Energy Distribution', fontsize=14)
        plt.xticks(x, band_labels, rotation=45)
        plt.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            return plt.gcf()
        
    @staticmethod
    def compute_publication_metrics(noisy_audio, denoised_audio, clean_audio=None, sr=16000):
        """
        Compute comprehensive metrics used in audio enhancement publications
        
        Args:
            noisy_audio: np.ndarray - Noisy input audio
            denoised_audio: np.ndarray - Denoised output audio
            clean_audio: np.ndarray or None - Clean reference audio (if available)
            sr: int - Sample rate
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {}
        
        # Make sure arrays have the same length
        min_len = min(len(noisy_audio), len(denoised_audio))
        noisy_audio = noisy_audio[:min_len]
        denoised_audio = denoised_audio[:min_len]
        
        if clean_audio is not None:
            clean_audio = clean_audio[:min_len]
        
        # 1. Signal-to-Noise Ratio (SNR) Estimation
        def estimate_snr(audio, noise_floor=None):
            """Estimate SNR using signal statistics"""
            if noise_floor is None:
                # Estimate noise floor from quietest 10% of frames
                frame_length = 2048
                hop_length = 512
                frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
                frame_energies = np.sum(frames**2, axis=0)
                sorted_energies = np.sort(frame_energies)
                noise_energy = np.mean(sorted_energies[:int(len(sorted_energies)*0.1)])
                signal_energy = np.mean(frame_energies) - noise_energy
                return 10 * np.log10(signal_energy / max(noise_energy, 1e-10))
            else:
                # If noise floor is provided (clean vs noisy case)
                noise = audio - noise_floor
                return 10 * np.log10(np.sum(noise_floor**2) / max(np.sum(noise**2), 1e-10))
        
        # Calculate estimated SNR
        noisy_snr = estimate_snr(noisy_audio)
        denoised_snr = estimate_snr(denoised_audio)
        metrics['estimated_snr_noisy'] = noisy_snr
        metrics['estimated_snr_denoised'] = denoised_snr
        metrics['snr_improvement'] = denoised_snr - noisy_snr
        
        # 2. Spectral Centroid (speech clarity/brightness metric)
        noisy_centroid = np.mean(librosa.feature.spectral_centroid(y=noisy_audio, sr=sr))
        denoised_centroid = np.mean(librosa.feature.spectral_centroid(y=denoised_audio, sr=sr))
        metrics['spectral_centroid_noisy'] = noisy_centroid
        metrics['spectral_centroid_denoised'] = denoised_centroid
        metrics['spectral_centroid_change'] = denoised_centroid - noisy_centroid
        
        # 3. Zero Crossing Rate (noise level indicator)
        noisy_zcr = np.mean(librosa.feature.zero_crossing_rate(y=noisy_audio))
        denoised_zcr = np.mean(librosa.feature.zero_crossing_rate(y=denoised_audio))
        metrics['zero_crossing_rate_noisy'] = noisy_zcr
        metrics['zero_crossing_rate_denoised'] = denoised_zcr
        metrics['zero_crossing_rate_reduction'] = noisy_zcr - denoised_zcr
        
        # 4. Spectral Contrast (speech vs noise contrast)
        noisy_contrast = np.mean(librosa.feature.spectral_contrast(y=noisy_audio, sr=sr))
        denoised_contrast = np.mean(librosa.feature.spectral_contrast(y=denoised_audio, sr=sr))
        metrics['spectral_contrast_noisy'] = noisy_contrast
        metrics['spectral_contrast_denoised'] = denoised_contrast
        metrics['spectral_contrast_improvement'] = denoised_contrast - noisy_contrast
        
        # 5. MFCC Distance (measure of speech distortion)
        noisy_mfcc = librosa.feature.mfcc(y=noisy_audio, sr=sr, n_mfcc=13)
        denoised_mfcc = librosa.feature.mfcc(y=denoised_audio, sr=sr, n_mfcc=13)
        mfcc_distance = np.mean(np.sqrt(np.sum((noisy_mfcc - denoised_mfcc)**2, axis=0)))
        metrics['mfcc_distance'] = mfcc_distance
        
        # 6. Spectral Flatness (measure of noisiness vs. tonality)
        noisy_flatness = np.mean(librosa.feature.spectral_flatness(y=noisy_audio))
        denoised_flatness = np.mean(librosa.feature.spectral_flatness(y=denoised_audio))
        metrics['spectral_flatness_noisy'] = noisy_flatness
        metrics['spectral_flatness_denoised'] = denoised_flatness
        metrics['spectral_flatness_reduction'] = noisy_flatness - denoised_flatness
        
        # 7. Dynamic Range
        def calculate_dynamic_range(audio):
            percentile_high = np.percentile(np.abs(audio), 95)
            percentile_low = np.percentile(np.abs(audio), 15)
            return 20 * np.log10(max(percentile_high / max(percentile_low, 1e-10), 1e-10))
        
        noisy_dr = calculate_dynamic_range(noisy_audio)
        denoised_dr = calculate_dynamic_range(denoised_audio)
        metrics['dynamic_range_db_noisy'] = noisy_dr
        metrics['dynamic_range_db_denoised'] = denoised_dr
        metrics['dynamic_range_change'] = denoised_dr - noisy_dr
        
        # 8. If clean reference is available, add reference-based metrics
        if clean_audio is not None:
            # Calculate true SNR
            true_noisy_snr = 10 * np.log10(np.sum(clean_audio**2) / np.sum((clean_audio - noisy_audio)**2))
            true_denoised_snr = 10 * np.log10(np.sum(clean_audio**2) / np.sum((clean_audio - denoised_audio)**2))
            metrics['true_snr_noisy'] = true_noisy_snr
            metrics['true_snr_denoised'] = true_denoised_snr
            metrics['true_snr_improvement'] = true_denoised_snr - true_noisy_snr
            
            # Try to calculate PESQ if available
            try:
                import pesq
                noisy_pesq = pesq.pesq(sr, clean_audio, noisy_audio, 'wb')
                denoised_pesq = pesq.pesq(sr, clean_audio, denoised_audio, 'wb')
                metrics['pesq_noisy'] = noisy_pesq
                metrics['pesq_denoised'] = denoised_pesq
                metrics['pesq_improvement'] = denoised_pesq - noisy_pesq
            except (ImportError, Exception) as e:
                print(f"Could not calculate PESQ: {e}")
            
            # Try to calculate STOI if available
            try:
                from pystoi import stoi
                noisy_stoi = stoi(clean_audio, noisy_audio, sr)
                denoised_stoi = stoi(clean_audio, denoised_audio, sr)
                metrics['stoi_noisy'] = noisy_stoi
                metrics['stoi_denoised'] = denoised_stoi
                metrics['stoi_improvement'] = denoised_stoi - noisy_stoi
            except (ImportError, Exception) as e:
                print(f"Could not calculate STOI: {e}")
        
        return metrics
        

class AblationStudy:
    """
    Class for conducting and visualizing ablation studies for the enhanced denoiser
    """
    
    @staticmethod
    def run_ablation_study(noisy_audio_path, clean_audio_path=None, sr=16000, output_dir=None):
        """
        Run ablation study on all enhancement techniques
        
        Args:
            noisy_audio_path: Path to noisy audio file
            clean_audio_path: Optional path to clean reference audio
            sr: Sample rate
            output_dir: Directory to save results
            
        Returns:
            dict: Results of ablation study
        """
        from denoiser.enhanced_denoiser import EnhancedDenoiserAudio
        import torch
        import os
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load audio files
        noisy_audio, sr = librosa.load(noisy_audio_path, sr=None)
        clean_audio = None
        if clean_audio_path:
            clean_audio, _ = librosa.load(clean_audio_path, sr=sr)
        
        # Define configurations for ablation
        configurations = {
            "all_features": {
                "adaptive_processing": True,
                "harmonic_enhancement": True,
                "vocal_clarity": True,
                "dynamic_range_compression": True,
                "domain_adaptation": True,
            },
            "no_adaptive": {
                "adaptive_processing": False,
                "harmonic_enhancement": True,
                "vocal_clarity": True,
                "dynamic_range_compression": True,
                "domain_adaptation": True,
            },
            "no_harmonic": {
                "adaptive_processing": True,
                "harmonic_enhancement": False,
                "vocal_clarity": True,
                "dynamic_range_compression": True,
                "domain_adaptation": True,
            },
            "no_vocal": {
                "adaptive_processing": True,
                "harmonic_enhancement": True,
                "vocal_clarity": False,
                "dynamic_range_compression": True,
                "domain_adaptation": True,
            },
            "no_compression": {
                "adaptive_processing": True,
                "harmonic_enhancement": True,
                "vocal_clarity": True,
                "dynamic_range_compression": False,
                "domain_adaptation": True,
            },
            "no_adaptation": {
                "adaptive_processing": True,
                "harmonic_enhancement": True,
                "vocal_clarity": True,
                "dynamic_range_compression": True,
                "domain_adaptation": False,
            },
            "baseline": {
                "adaptive_processing": False,
                "harmonic_enhancement": False,
                "vocal_clarity": False,
                "dynamic_range_compression": False,
                "domain_adaptation": False,
            }
        }
        
        results = {}
        enhanced_audios = {}
        
        # Process with each configuration
        for config_name, config in configurations.items():
            print(f"Processing configuration: {config_name}")
            
            # Create denoiser with this configuration
            denoiser = EnhancedDenoiserAudio(
                device=device,
                chunk_length_s=3,
                max_batch_size=20,
                adaptive_processing=config["adaptive_processing"],
                harmonic_enhancement=config["harmonic_enhancement"],
                vocal_clarity=config["vocal_clarity"],
                dynamic_range_compression=config["dynamic_range_compression"],
                domain_adaptation=config["domain_adaptation"],
                verbose=True
            )
            
            # Process audio
            output_path = os.path.join(output_dir, f"{config_name}.wav") if output_dir else None
            enhanced_audio, _ = denoiser(noisy_audio_path, output_path)
            enhanced_audios[config_name] = enhanced_audio
            
            # Compute metrics if clean reference is available
            if clean_audio is not None:
                metrics = AudioMetrics.compute_metrics(clean_audio, noisy_audio, enhanced_audio, sr)
                results[config_name] = metrics
        
        # Generate ablation study visualization
        if output_dir:
            AblationStudy.visualize_ablation_results(results, os.path.join(output_dir, "ablation_study.png"))
            
            # Generate comparison spectrograms
            AblationStudy.plot_ablation_spectrograms(
                enhanced_audios, 
                sr=sr, 
                output_path=os.path.join(output_dir, "ablation_spectrograms.png")
            )
        
        return results, enhanced_audios
    
    @staticmethod
    def visualize_ablation_results(results, output_path=None, fig_size=(16, 10)):
        """
        Visualize results of ablation study
        """
        plt.figure(figsize=fig_size)
        
        # Extract key metrics
        configs = list(results.keys())
        snr_improvements = [results[c]['snr_improvement'] for c in configs]
        pesq_improvements = [results[c]['pesq_improvement'] for c in configs]
        stoi_improvements = [results[c]['stoi_improvement'] for c in configs]
        
        # Normalize metrics for radar chart
        def normalize(values):
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                return [0.5 for _ in values]
            return [(v - min_val) / (max_val - min_val) for v in values]
        
        snr_norm = normalize(snr_improvements)
        pesq_norm = normalize(pesq_improvements)
        stoi_norm = normalize(stoi_improvements)
        
        # 1. Bar chart for SNR improvements
        plt.subplot(2, 2, 1)
        x = np.arange(len(configs))
        plt.bar(x, snr_improvements)
        plt.xlabel('Configuration')
        plt.ylabel('SNR Improvement (dB)')
        plt.title('Effect on SNR Improvement')
        plt.xticks(x, configs, rotation=45)
        
        # 2. Bar chart for PESQ improvements
        plt.subplot(2, 2, 2)
        plt.bar(x, pesq_improvements)
        plt.xlabel('Configuration')
        plt.ylabel('PESQ Improvement')
        plt.title('Effect on PESQ Improvement')
        plt.xticks(x, configs, rotation=45)
        
        # 3. Bar chart for STOI improvements
        plt.subplot(2, 2, 3)
        plt.bar(x, stoi_improvements)
        plt.xlabel('Configuration')
        plt.ylabel('STOI Improvement')
        plt.title('Effect on STOI Improvement')
        plt.xticks(x, configs, rotation=45)
        
        # 4. Contribution table
        plt.subplot(2, 2, 4)
        plt.axis('off')
        
        # Calculate contribution of each feature
        baseline = results['baseline']
        all_features = results['all_features']
        
        contribution_text = "Feature Contributions:\n\n"
        
        for feature in ['adaptive', 'harmonic', 'vocal', 'compression', 'adaptation']:
            config_name = f"no_{feature}"
            if config_name in results:
                # Calculate contribution as percentage of total improvement
                snr_contrib = ((all_features['snr_improvement'] - results[config_name]['snr_improvement']) / 
                              (all_features['snr_improvement'] - baseline['snr_improvement']) * 100)
                
                pesq_contrib = ((all_features['pesq_improvement'] - results[config_name]['pesq_improvement']) / 
                               (all_features['pesq_improvement'] - baseline['pesq_improvement']) * 100)
                
                stoi_contrib = ((all_features['stoi_improvement'] - results[config_name]['stoi_improvement']) / 
                               (all_features['stoi_improvement'] - baseline['stoi_improvement']) * 100)
                
                feature_name = " ".join(word.capitalize() for word in feature.split("_"))
                contribution_text += f"{feature_name}:\n"
                contribution_text += f"  SNR: {snr_contrib:.1f}%\n"
                contribution_text += f"  PESQ: {pesq_contrib:.1f}%\n"
                contribution_text += f"  STOI: {stoi_contrib:.1f}%\n\n"
        
        plt.text(0.1, 0.1, contribution_text, fontsize=10, va='top')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            return plt.gcf()
    
    @staticmethod
    def plot_ablation_spectrograms(enhanced_audios, sr=16000, output_path=None, fig_size=(20, 15)):
        """
        Plot spectrograms for ablation study
        """
        configs = list(enhanced_audios.keys())
        n_configs = len(configs)
        
        # Calculate grid size
        n_cols = 2
        n_rows = (n_configs + 1) // n_cols  # +1 to ensure we have enough rows
        
        plt.figure(figsize=fig_size)
        
        for i, config in enumerate(configs):
            plt.subplot(n_rows, n_cols, i+1)
            
            # Compute spectrogram
            audio = enhanced_audios[config]
            S = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            
            # Plot spectrogram
            img = librosa.display.specshow(
                S, y_axis='log', x_axis='time', sr=sr, cmap='viridis'
            )
            
            plt.title(f"{config} Spectrogram")
        
        plt.colorbar(img, format='%+2.0f dB', ax=plt.gcf().axes[-1])
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            return output_path
        else:
            return plt.gcf()