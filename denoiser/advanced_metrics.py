import numpy as np
import librosa
import scipy.signal as signal
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt

class AdvancedAudioMetrics:
    """
    Enhanced audio metrics calculation for research paper comparisons
    """
    
    @staticmethod
    def calculate_harmonic_to_noise_ratio(audio: np.ndarray, sr: int = 16000, frame_length: int = 2048, hop_length: int = 512) -> float:
        """
        Calculate Harmonic-to-Noise Ratio (HNR) of audio signal
        
        Args:
            audio: Audio signal
            sr: Sample rate
            frame_length: Frame length for analysis
            hop_length: Hop length between frames
            
        Returns:
            float: Harmonic-to-Noise Ratio in dB
        """
        try:
            # Extract pitch
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr,
                frame_length=frame_length,
                hop_length=hop_length
            )
            
            # Only process voiced segments
            voiced_frames = np.where(voiced_flag)[0]
            
            if len(voiced_frames) == 0:
                return 0.0
            
            # Calculate HNR
            hnr_values = []
            
            for i in voiced_frames:
                start_sample = i * hop_length
                end_sample = start_sample + frame_length
                
                if end_sample > len(audio):
                    continue
                    
                frame = audio[start_sample:end_sample]
                
                # Apply window
                window = np.hanning(len(frame))
                frame = frame * window
                
                # Skip empty frames
                if np.sum(frame ** 2) < 1e-10:
                    continue
                
                # Get pitch for this frame
                if np.isnan(f0[i]):
                    continue
                    
                pitch = f0[i]
                
                # Create synthetic harmonic signal
                period = int(sr / pitch)
                if period < 2:  # Too high pitch for reliable analysis
                    continue
                    
                n_periods = max(1, len(frame) // period)
                
                # Prototype period extraction
                prototype = np.zeros(period)
                count = np.zeros(period)
                
                for j in range(n_periods):
                    start_j = j * period
                    end_j = start_j + period
                    
                    if end_j > len(frame):
                        break
                        
                    prototype += frame[start_j:end_j]
                    count += 1
                
                # Avoid division by zero
                count[count == 0] = 1
                prototype /= count
                
                # Replicate prototype to create harmonic signal
                harmonic = np.tile(prototype, n_periods)[:len(frame)]
                
                # Calculate residual (noise)
                residual = frame - harmonic
                
                # Calculate HNR
                harmonic_power = np.sum(harmonic ** 2)
                noise_power = np.sum(residual ** 2)
                
                if noise_power > 0:
                    hnr = 10 * np.log10(harmonic_power / noise_power)
                    hnr_values.append(hnr)
            
            # Return average HNR
            if len(hnr_values) > 0:
                return np.mean(hnr_values)
            else:
                return 0.0
                
        except Exception as e:
            print(f"Error calculating HNR: {e}")
            return 0.0
    
    @staticmethod
    def calculate_segmental_snr(clean: np.ndarray, processed: np.ndarray, sr: int = 16000,
                               frame_length: int = 1024, hop_length: int = 512) -> float:
        """
        Calculate segmental SNR (Signal-to-Noise Ratio)
        
        Args:
            clean: Clean reference audio
            processed: Processed audio
            sr: Sample rate
            frame_length: Frame length in samples
            hop_length: Hop length between frames
            
        Returns:
            float: Segmental SNR in dB
        """
        # Make sure both signals have the same length
        min_len = min(len(clean), len(processed))
        clean = clean[:min_len]
        processed = processed[:min_len]
        
        # Frame-wise processing
        n_frames = 1 + (min_len - frame_length) // hop_length
        snr_values = []
        
        for i in range(n_frames):
            start = i * hop_length
            end = start + frame_length
            
            # Get frame
            clean_frame = clean[start:end]
            processed_frame = processed[start:end]
            
            # Calculate noise
            noise_frame = clean_frame - processed_frame
            
            # Calculate frame SNR
            signal_power = np.sum(clean_frame ** 2)
            noise_power = np.sum(noise_frame ** 2)
            
            if noise_power > 0 and signal_power > 0:
                snr = 10 * np.log10(signal_power / noise_power)
                # Clamp SNR to reasonable range
                snr = max(-10, min(35, snr))
                snr_values.append(snr)
        
        # Return average segmental SNR
        if len(snr_values) > 0:
            return np.mean(snr_values)
        else:
            return 0.0
    
    @staticmethod
    def calculate_spectral_distance(clean: np.ndarray, processed: np.ndarray, sr: int = 16000,
                                   frame_length: int = 1024, hop_length: int = 512) -> float:
        """
        Calculate spectral distance using Log-Spectral Distance (LSD)
        
        Args:
            clean: Clean reference audio
            processed: Processed audio
            sr: Sample rate
            frame_length: Frame length in samples
            hop_length: Hop length between frames
            
        Returns:
            float: Log-Spectral Distance (lower is better)
        """
        # Make sure both signals have the same length
        min_len = min(len(clean), len(processed))
        clean = clean[:min_len]
        processed = processed[:min_len]
        
        # Calculate spectrograms
        clean_spec = np.abs(librosa.stft(clean, n_fft=frame_length, hop_length=hop_length))
        processed_spec = np.abs(librosa.stft(processed, n_fft=frame_length, hop_length=hop_length))
        
        # Avoid log of zero
        eps = 1e-8
        clean_log_spec = np.log10(np.maximum(eps, clean_spec))
        processed_log_spec = np.log10(np.maximum(eps, processed_spec))
        
        # Calculate distance
        diff = clean_log_spec - processed_log_spec
        squared_diff = diff ** 2
        
        # Mean over frequency bands for each frame
        frame_distances = np.sqrt(np.mean(squared_diff, axis=0))
        
        # Return average distance
        return np.mean(frame_distances)
    
    @staticmethod
    def calculate_speech_intelligibility(clean: np.ndarray, processed: np.ndarray, sr: int = 16000) -> float:
        """
        Calculate speech intelligibility using a simplified approach
        
        Args:
            clean: Clean reference audio
            processed: Processed audio
            sr: Sample rate
            
        Returns:
            float: Speech intelligibility score (higher is better)
        """
        # Make sure both signals have the same length
        min_len = min(len(clean), len(processed))
        clean = clean[:min_len]
        processed = processed[:min_len]
        
        # Define speech-important frequency bands (approximation)
        # Critical bands for speech intelligibility: 500-4000 Hz
        lower_freq = 500  # Hz
        upper_freq = 4000  # Hz
        
        # Convert to FFT bin indices
        n_fft = 2048
        fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        lower_bin = np.argmin(np.abs(fft_freqs - lower_freq))
        upper_bin = np.argmin(np.abs(fft_freqs - upper_freq))
        
        # Calculate FFT
        clean_fft = np.abs(librosa.stft(clean, n_fft=n_fft))
        processed_fft = np.abs(librosa.stft(processed, n_fft=n_fft))
        
        # Focus on speech-important bands
        clean_speech_bands = clean_fft[lower_bin:upper_bin, :]
        processed_speech_bands = processed_fft[lower_bin:upper_bin, :]
        
        # Calculate correlation
        correlations = []
        for i in range(clean_speech_bands.shape[1]):
            corr = np.corrcoef(clean_speech_bands[:, i], processed_speech_bands[:, i])[0, 1]
            if not np.isnan(corr):
                correlations.append(corr)
        
        # Return average correlation as intelligibility score
        if len(correlations) > 0:
            return np.mean(correlations)
        else:
            return 0.0
    
    @staticmethod
    def calculate_comprehensive_metrics(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray, 
                                        sr: int = 16000) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics for audio quality assessment
        
        Args:
            clean: Clean reference audio
            noisy: Noisy input audio
            denoised: Denoised output audio
            sr: Sample rate
            
        Returns:
            Dict: Comprehensive metrics dictionary
        """
        # Ensure all signals have the same length
        min_len = min(len(clean), len(noisy), len(denoised))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        denoised = denoised[:min_len]
        
        # Calculate basic metrics
        metrics = {}
        
        # SNR metrics
        metrics["noisy_snr"] = 10 * np.log10(np.sum(clean**2) / np.sum((clean - noisy)**2 + 1e-10))
        metrics["denoised_snr"] = 10 * np.log10(np.sum(clean**2) / np.sum((clean - denoised)**2 + 1e-10))
        metrics["snr_improvement"] = metrics["denoised_snr"] - metrics["noisy_snr"]
        
        # Segmental SNR
        metrics["noisy_segsnr"] = AdvancedAudioMetrics.calculate_segmental_snr(clean, noisy, sr)
        metrics["denoised_segsnr"] = AdvancedAudioMetrics.calculate_segmental_snr(clean, denoised, sr)
        metrics["segsnr_improvement"] = metrics["denoised_segsnr"] - metrics["noisy_segsnr"]
        
        # Harmonic-to-Noise Ratio
        metrics["clean_hnr"] = AdvancedAudioMetrics.calculate_harmonic_to_noise_ratio(clean, sr)
        metrics["noisy_hnr"] = AdvancedAudioMetrics.calculate_harmonic_to_noise_ratio(noisy, sr)
        metrics["denoised_hnr"] = AdvancedAudioMetrics.calculate_harmonic_to_noise_ratio(denoised, sr)
        metrics["hnr_improvement"] = metrics["denoised_hnr"] - metrics["noisy_hnr"]
        metrics["hnr_restoration"] = (metrics["denoised_hnr"] - metrics["noisy_hnr"]) / (metrics["clean_hnr"] - metrics["noisy_hnr"] + 1e-10)
        
        # Spectral distance
        metrics["noisy_lsd"] = AdvancedAudioMetrics.calculate_spectral_distance(clean, noisy, sr)
        metrics["denoised_lsd"] = AdvancedAudioMetrics.calculate_spectral_distance(clean, denoised, sr)
        metrics["lsd_improvement"] = metrics["noisy_lsd"] - metrics["denoised_lsd"]  # Lower is better
        
        # Speech intelligibility
        metrics["noisy_intel"] = AdvancedAudioMetrics.calculate_speech_intelligibility(clean, noisy, sr)
        metrics["denoised_intel"] = AdvancedAudioMetrics.calculate_speech_intelligibility(clean, denoised, sr)
        metrics["intel_improvement"] = metrics["denoised_intel"] - metrics["noisy_intel"]
        
        return metrics
    
    @staticmethod
    def plot_comparative_metrics(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray, 
                              baseline_denoised: Optional[np.ndarray] = None,
                              sr: int = 16000, output_path: Optional[str] = None) -> plt.Figure:
        """
        Generate comparative metrics visualization
        
        Args:
            clean: Clean reference audio
            noisy: Noisy input audio
            denoised: Denoised output audio
            baseline_denoised: Optional baseline model output for comparison
            sr: Sample rate
            output_path: Optional path to save the figure
            
        Returns:
            plt.Figure: Matplotlib figure with comparative metrics
        """
        # Calculate metrics for enhanced model
        enhanced_metrics = AdvancedAudioMetrics.calculate_comprehensive_metrics(
            clean, noisy, denoised, sr
        )
        
        # Calculate metrics for baseline model if provided
        baseline_metrics = None
        if baseline_denoised is not None:
            baseline_metrics = AdvancedAudioMetrics.calculate_comprehensive_metrics(
                clean, noisy, baseline_denoised, sr
            )
        
        # Set up figure
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. SNR and Segmental SNR plot (top left)
        ax = axs[0, 0]
        labels = ['SNR (dB)', 'Segmental SNR (dB)']
        x = np.arange(len(labels))
        width = 0.25
        
        noisy_values = [enhanced_metrics["noisy_snr"], enhanced_metrics["noisy_segsnr"]]
        enhanced_values = [enhanced_metrics["denoised_snr"], enhanced_metrics["denoised_segsnr"]]
        
        bars1 = ax.bar(x - width, noisy_values, width, label='Noisy', color='#EA4335')
        bars2 = ax.bar(x, enhanced_values, width, label='Enhanced', color='#4285F4')
        
        if baseline_metrics:
            baseline_values = [baseline_metrics["denoised_snr"], baseline_metrics["denoised_segsnr"]]
            bars3 = ax.bar(x + width, baseline_values, width, label='Baseline', color='#FBBC05')
        
        ax.set_ylabel('dB')
        ax.set_title('Signal-to-Noise Ratio')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        # Add value labels
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom')
                
        add_labels(bars1)
        add_labels(bars2)
        if baseline_metrics:
            add_labels(bars3)
        
        # 2. HNR and Intelligibility plot (top right)
        ax = axs[0, 1]
        labels = ['HNR (dB)', 'Speech Intelligibility']
        x = np.arange(len(labels))
        width = 0.25
        
        noisy_values = [enhanced_metrics["noisy_hnr"], enhanced_metrics["noisy_intel"]]
        enhanced_values = [enhanced_metrics["denoised_hnr"], enhanced_metrics["denoised_intel"]]
        
        bars1 = ax.bar(x - width, noisy_values, width, label='Noisy', color='#EA4335')
        bars2 = ax.bar(x, enhanced_values, width, label='Enhanced', color='#4285F4')
        
        if baseline_metrics:
            baseline_values = [baseline_metrics["denoised_hnr"], baseline_metrics["denoised_intel"]]
            bars3 = ax.bar(x + width, baseline_values, width, label='Baseline', color='#FBBC05')
        
        ax.set_ylabel('Value')
        ax.set_title('Harmonic Quality and Intelligibility')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        
        add_labels(bars1)
        add_labels(bars2)
        if baseline_metrics:
            add_labels(bars3)
        
        # 3. Improvements plot (bottom left)
        ax = axs[1, 0]
        labels = ['SNR', 'Seg. SNR', 'HNR', 'LSD', 'Intelligibility']
        x = np.arange(len(labels))
        
        # For LSD lower is better, so improvement is negative
        enhanced_improvements = [
            enhanced_metrics["snr_improvement"],
            enhanced_metrics["segsnr_improvement"],
            enhanced_metrics["hnr_improvement"],
            enhanced_metrics["lsd_improvement"] * (-1),  # Reverse sign for visualization
            enhanced_metrics["intel_improvement"] * 100  # Scale up for visibility
        ]
        
        bars = ax.bar(x, enhanced_improvements, 
                      color=['#4285F4' if val >= 0 else '#EA4335' for val in enhanced_improvements])
        
        ax.set_ylabel('Improvement')
        ax.set_title('Enhanced Model Improvements')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add value labels with appropriate scaling
        for i, bar in enumerate(bars):
            height = bar.get_height()
            if i == 4:  # Intelligibility was scaled up
                label = f'{height/100:.3f}'
            else:
                label = f'{height:.2f}'
            
            ax.annotate(label,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3 if height >= 0 else -14),
                       textcoords="offset points",
                       ha='center', va='bottom' if height >= 0 else 'top')
        
        # 4. Comparison with baseline if available (bottom right)
        ax = axs[1, 1]
        
        if baseline_metrics:
            labels = ['SNR', 'Seg. SNR', 'HNR', 'LSD', 'Intelligibility']
            x = np.arange(len(labels))
            width = 0.4
            
            enhanced_improvements = [
                enhanced_metrics["snr_improvement"],
                enhanced_metrics["segsnr_improvement"],
                enhanced_metrics["hnr_improvement"],
                enhanced_metrics["lsd_improvement"],
                enhanced_metrics["intel_improvement"]
            ]
            
            baseline_improvements = [
                baseline_metrics["snr_improvement"],
                baseline_metrics["segsnr_improvement"],
                baseline_metrics["hnr_improvement"],
                baseline_metrics["lsd_improvement"],
                baseline_metrics["intel_improvement"]
            ]
            
            # Calculate relative improvement over baseline
            relative_improvement = [
                (enhanced - baseline) / max(abs(baseline), 1e-10) * 100
                for enhanced, baseline in zip(enhanced_improvements, baseline_improvements)
            ]
            
            bars = ax.bar(x, relative_improvement,
                         color=['#4285F4' if val >= 0 else '#EA4335' for val in relative_improvement])
            
            ax.set_ylabel('Relative Improvement (%)')
            ax.set_title('Enhanced vs. Baseline Model')
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -14),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top')
        else:
            # If no baseline, show HNR restoration rate
            ax.remove()
            ax = fig.add_subplot(2, 2, 4)
            
            # Create a simple pie chart showing HNR restoration percentage
            hnr_restoration = enhanced_metrics["hnr_restoration"] * 100
            hnr_restoration = max(0, min(100, hnr_restoration))  # Clamp to [0,100]
            
            sizes = [hnr_restoration, 100 - hnr_restoration]
            labels = ['Restored', 'Not Restored']
            colors = ['#4285F4', '#E0E0E0']
            explode = (0.1, 0)  # explode the first slice
            
            ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%',
                  shadow=True, startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular
            ax.set_title('Harmonic Structure Restoration')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    @staticmethod
    def plot_spectrogram_comparison_enhanced(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray,
                                          sr: int = 16000, output_path: Optional[str] = None) -> plt.Figure:
        """
        Generate an enhanced spectrogram comparison visualization with detailed frequency analysis
        
        Args:
            clean: Clean reference audio
            noisy: Noisy input audio
            denoised: Denoised output audio
            sr: Sample rate
            output_path: Optional path to save the figure
            
        Returns:
            plt.Figure: Matplotlib figure with spectrogram comparison
        """
        # Ensure all signals have the same length
        min_len = min(len(clean), len(noisy), len(denoised))
        clean = clean[:min_len]
        noisy = noisy[:min_len]
        denoised = denoised[:min_len]
        
        # Set up the plot
        fig = plt.figure(figsize=(15, 12))
        gs = fig.add_gridspec(3, 3)
        
        # Main spectrograms
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        
        # Difference spectrograms
        ax4 = fig.add_subplot(gs[1, 0])
        ax5 = fig.add_subplot(gs[1, 1])
        ax6 = fig.add_subplot(gs[1, 2])
        
        # Frequency analysis
        ax7 = fig.add_subplot(gs[2, :])
        
        # Compute spectrograms
        n_fft = 1024
        hop_length = 256
        
        # Clean spectrogram
        S_clean = librosa.stft(clean, n_fft=n_fft, hop_length=hop_length)
        D_clean = librosa.amplitude_to_db(np.abs(S_clean), ref=np.max)
        
        # Noisy spectrogram
        S_noisy = librosa.stft(noisy, n_fft=n_fft, hop_length=hop_length)
        D_noisy = librosa.amplitude_to_db(np.abs(S_noisy), ref=np.max)
        
        # Denoised spectrogram
        S_denoised = librosa.stft(denoised, n_fft=n_fft, hop_length=hop_length)
        D_denoised = librosa.amplitude_to_db(np.abs(S_denoised), ref=np.max)
        
        # Noise difference (noisy - clean)
        D_noise = D_noisy - D_clean
        
        # Denoising difference (denoised - clean)
        D_denoise_diff = D_denoised - D_clean
        
        # Improvement (noisy vs denoised differences)
        D_improvement = np.abs(D_noise) - np.abs(D_denoise_diff)
        
        # Set common color scale for main spectrograms
        vmin = min(np.min(D_clean), np.min(D_noisy), np.min(D_denoised))
        vmax = max(np.max(D_clean), np.max(D_noisy), np.max(D_denoised))
        
        # Plot spectrograms
        img1 = librosa.display.specshow(D_clean, sr=sr, hop_length=hop_length, 
                                     x_axis='time', y_axis='log', ax=ax1, vmin=vmin, vmax=vmax)
        ax1.set_title('Clean Audio')
        
        img2 = librosa.display.specshow(D_noisy, sr=sr, hop_length=hop_length, 
                                     x_axis='time', y_axis='log', ax=ax2, vmin=vmin, vmax=vmax)
        ax2.set_title('Noisy Audio')
        
        img3 = librosa.display.specshow(D_denoised, sr=sr, hop_length=hop_length, 
                                     x_axis='time', y_axis='log', ax=ax3, vmin=vmin, vmax=vmax)
        ax3.set_title('Denoised Audio')
        
        # Add colorbar for main spectrograms
        fig.colorbar(img3, ax=[ax1, ax2, ax3], format='%+2.0f dB')
        
        # Plot difference spectrograms with diverging colormap
        diff_vmax = max(np.max(np.abs(D_noise)), np.max(np.abs(D_denoise_diff)), np.max(np.abs(D_improvement)))
        
        img4 = librosa.display.specshow(D_noise, sr=sr, hop_length=hop_length, 
                                     x_axis='time', y_axis='log', ax=ax4, 
                                     cmap='coolwarm', vmin=-diff_vmax, vmax=diff_vmax)
        ax4.set_title('Noise Difference (Noisy - Clean)')
        
        img5 = librosa.display.specshow(D_denoise_diff, sr=sr, hop_length=hop_length, 
                                     x_axis='time', y_axis='log', ax=ax5, 
                                     cmap='coolwarm', vmin=-diff_vmax, vmax=diff_vmax)
        ax5.set_title('Denoising Residual (Denoised - Clean)')
        
        img6 = librosa.display.specshow(D_improvement, sr=sr, hop_length=hop_length, 
                                     x_axis='time', y_axis='log', ax=ax6, 
                                     cmap='PiYG', vmin=-diff_vmax, vmax=diff_vmax)
        ax6.set_title('Improvement Map')
        
        # Add colorbar for difference spectrograms
        fig.colorbar(img4, ax=[ax4, ax5, ax6], format='%+2.0f dB')
        
        # Frequency analysis
        # Calculate average power in frequency bands
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        
        clean_power = np.mean(np.abs(S_clean)**2, axis=1)
        noisy_power = np.mean(np.abs(S_noisy)**2, axis=1)
        denoised_power = np.mean(np.abs(S_denoised)**2, axis=1)
        
        # Convert to dB
        clean_power_db = 10 * np.log10(clean_power + 1e-10)
        noisy_power_db = 10 * np.log10(noisy_power + 1e-10)
        denoised_power_db = 10 * np.log10(denoised_power + 1e-10)
        
        # Smooth the curves for better visualization
        def smooth(y, box_pts):
            box = np.ones(box_pts) / box_pts
            y_smooth = np.convolve(y, box, mode='same')
            return y_smooth
            
        smoothing = max(1, len(freqs) // 100)
        clean_power_smooth = smooth(clean_power_db, smoothing)
        noisy_power_smooth = smooth(noisy_power_db, smoothing)
        denoised_power_smooth = smooth(denoised_power_db, smoothing)
        
        # Plot frequency analysis
        ax7.semilogx(freqs, clean_power_smooth, label='Clean', alpha=0.8, linewidth=2, color='#34A853')
        ax7.semilogx(freqs, noisy_power_smooth, label='Noisy', alpha=0.8, linewidth=2, color='#EA4335')
        ax7.semilogx(freqs, denoised_power_smooth, label='Denoised', alpha=0.8, linewidth=2, color='#4285F4')
        
        # Add shaded areas for speech frequencies
        speech_freqs = [(250, 500, 'Vowels'), (500, 2000, 'Formants'), (2000, 4000, 'Consonants')]
        colors = ['rgba(52, 168, 83, 0.2)', 'rgba(66, 133, 244, 0.2)', 'rgba(251, 188, 5, 0.2)']
        
        y_min, y_max = ax7.get_ylim()
        
        for i, (low, high, label) in enumerate(speech_freqs):
            ax7.axvspan(low, high, alpha=0.2, color=colors[i % len(colors)], 
                       label=f'{label} ({low}-{high} Hz)')
            
        ax7.set_xlim([20, sr/2])
        ax7.set_xlabel('Frequency (Hz)')
        ax7.set_ylabel('Power (dB)')
        ax7.set_title('Frequency Content Analysis')
        ax7.legend()
        ax7.grid(True, which='both', linestyle='--', alpha=0.3)
        
        # Calculate and annotate speech SNR in critical bands
        for low, high, label in speech_freqs:
            # Find frequency band indices
            band_indices = np.where((freqs >= low) & (freqs <= high))[0]
            
            if len(band_indices) > 0:
                # Calculate band power
                clean_band_power = np.mean(np.abs(S_clean[band_indices, :])**2)
                noisy_band_power = np.mean(np.abs(S_noisy[band_indices, :])**2)
                denoised_band_power = np.mean(np.abs(S_denoised[band_indices, :])**2)
                
                # Calculate noise power
                noisy_noise_power = noisy_band_power - clean_band_power
                denoised_noise_power = denoised_band_power - clean_band_power
                
                # Calculate SNR
                noisy_band_snr = 10 * np.log10(clean_band_power / max(noisy_noise_power, 1e-10))
                denoised_band_snr = 10 * np.log10(clean_band_power / max(denoised_noise_power, 1e-10))
                
                # Annotate
                mid_freq = np.exp((np.log(low) + np.log(high)) / 2)
                y_pos = y_min + 0.8 * (y_max - y_min)
                ax7.annotate(f'SNR: {noisy_band_snr:.1f}→{denoised_band_snr:.1f} dB',
                            xy=(mid_freq, y_pos),
                            xytext=(0, 0),
                            textcoords="offset points",
                            ha='center', va='center',
                            bbox=dict(boxstyle="round,pad=0.3", alpha=0.2, fc='white'))
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    @staticmethod
    def generate_research_metrics_report(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray,
                                         sr: int = 16000, output_dir: str = 'research_metrics') -> Dict[str, Any]:
        """
        Generate a comprehensive research metrics report
        
        Args:
            clean: Clean reference audio
            noisy: Noisy input audio
            denoised: Denoised output audio
            sr: Sample rate
            output_dir: Directory to save outputs
            
        Returns:
            Dict: Metrics data and file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate comprehensive metrics
        metrics = AdvancedAudioMetrics.calculate_comprehensive_metrics(clean, noisy, denoised, sr)
        
        # Save metrics as JSON
        metrics_path = os.path.join(output_dir, 'advanced_metrics.json')
        with open(metrics_path, 'w') as f:
            import json
            json.dump(metrics, f, indent=4)
        
        # Generate comparative metrics visualization
        metrics_chart_path = os.path.join(output_dir, 'advanced_metrics_chart.png')
        AdvancedAudioMetrics.plot_comparative_metrics(
            clean, noisy, denoised, sr=sr, output_path=metrics_chart_path
        )
        
        # Generate spectrogram visualization
        spec_chart_path = os.path.join(output_dir, 'enhanced_spectrogram_analysis.png')
        AdvancedAudioMetrics.plot_spectrogram_comparison_enhanced(
            clean, noisy, denoised, sr=sr, output_path=spec_chart_path
        )
        
        return {
            'metrics': metrics,
            'metrics_json': metrics_path,
            'metrics_chart': metrics_chart_path,
            'spectrogram_chart': spec_chart_path
        }