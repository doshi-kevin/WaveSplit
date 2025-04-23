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