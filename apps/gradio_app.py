import gradio as gr
import os
import torch
import torchaudio
import tempfile
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import librosa
import librosa.display
from typing import Tuple, Optional, List, Dict, Any, Union
import json
import time

# Import the enhanced denoiser instead of the original one
from denoiser.enhanced_denoiser import EnhancedDenoiserAudio
from visualization_utils import AudioMetrics

# Initialize enhanced denoiser with proper settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
denoise = EnhancedDenoiserAudio(
    device=device,
    chunk_length_s=3,
    max_batch_size=20,
    adaptive_processing=True,
    harmonic_enhancement=True,
    vocal_clarity=True,
    dynamic_range_compression=False,
    domain_adaptation=False,
    verbose=True
)

def process_audio(audio, 
                  use_adaptive: bool = True, 
                  use_harmonic: bool = True, 
                  use_vocal: bool = True,
                  use_compression: bool = False,
                  progress=gr.Progress()):
    """
    Process uploaded audio to denoise it with enhanced features.
    
    Args:
        audio: Input audio file path or tuple of (sample_rate, audio_data)
        use_adaptive: Whether to use adaptive SNR processing
        use_harmonic: Whether to use harmonic enhancement
        use_vocal: Whether to use vocal clarity enhancement
        use_compression: Whether to use dynamic range compression
        progress: Gradio progress tracker
        
    Returns:
        tuple: Denoised audio file, spectrogram image, metrics HTML, status message
    """
    if audio is None:
        return None, None, None, "⚠️ No audio file uploaded. Please upload an audio file."
    
    # Create temp directory if it doesn't exist
    os.makedirs("Temp", exist_ok=True)
    os.makedirs("Temp/Metrics", exist_ok=True)
    
    progress(0, "Preparing audio file...")
    
    temp_filepath = None
    original_filepath = None
    
    try:
        # Handle different input types
        if isinstance(audio, tuple):  # Recorded audio
            sample_rate, audio_data = audio
            
            # Print debug info
            print(f"Received audio: sample_rate={sample_rate}, audio_data shape={np.array(audio_data).shape}, type={type(audio_data)}")
            
            # Create a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav', dir="Temp")
            temp_filepath = temp_file.name
            temp_file.close()
            
            # Ensure audio_data is in the right format
            audio_np = np.array(audio_data)
            
            # Use soundfile for more robust saving
            try:
                # Normalize audio data if needed
                if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                    max_val = max(abs(audio_np.max()), abs(audio_np.min()))
                    audio_np = audio_np / max_val
                
                # Write audio data to file
                sf.write(temp_filepath, audio_np, sample_rate)
                original_filepath = temp_filepath
                print(f"Successfully saved audio to {temp_filepath}")
                
            except Exception as save_error:
                print(f"Error with soundfile: {save_error}")
                # Fall back to torchaudio if soundfile fails
                audio_tensor = torch.tensor(audio_np)
                if audio_tensor.dim() == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                torchaudio.save(temp_filepath, audio_tensor, sample_rate)
                original_filepath = temp_filepath
                print(f"Successfully saved audio with torchaudio to {temp_filepath}")
                
        else:  # Uploaded file
            original_filepath = audio
            print(f"Using uploaded file: {original_filepath}")
        
        # Generate output filename
        filename = os.path.basename(original_filepath)
        denoised_filepath = os.path.join("Temp", f"denoised_{filename}")
        metrics_dir = os.path.join("Temp/Metrics", f"metrics_{int(time.time())}")
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Create spectrogram of original audio
        progress(0.1, "Generating original spectrogram...")
        orig_audio, sr = librosa.load(original_filepath, sr=None)
        
        # Update denoiser settings
        denoise.adaptive_processing = use_adaptive
        denoise.harmonic_enhancement = use_harmonic
        denoise.vocal_clarity = use_vocal
        denoise.dynamic_range_compression = use_compression
        
        # Process audio with description of enabled features
        progress(0.2, f"Processing audio with advanced denoising techniques...")
        
        # Denoise the audio
        denoised_audio, metrics_data = denoise(
            noisy_audio_path=original_filepath,
            output_path=denoised_filepath,
            metrics_path=metrics_dir
        )
        
        progress(0.7, "Generating spectrograms...")
        
        # Load denoised audio for visualization
        denoised_audio_vis, _ = librosa.load(denoised_filepath, sr=None)
        
        # Generate spectrograms
        spectrograms_path = os.path.join(metrics_dir, "spectrograms.png")
        AudioMetrics.plot_spectrograms(
            noisy_audio=orig_audio,
            denoised_audio=denoised_audio_vis,
            sr=sr,
            output_path=spectrograms_path
        )
        
        progress(0.8, "Generating metrics visualization...")
        
        # Create metrics HTML
        metrics_html = generate_metrics_html(metrics_data)
        metrics_html_path = os.path.join(metrics_dir, "metrics.html")
        with open(metrics_html_path, "w", encoding="utf-8") as f:
            f.write(metrics_html)

        path_for_url = metrics_html_path.replace("\\", "/")
        iframe_html = f'<iframe src="file:///{path_for_url}" width="100%" height="500px" frameborder="0"></iframe>'
                
        
        
        progress(0.9, "Finalizing...")
        
        # Clean up temporary file if it was created
        if temp_filepath and os.path.exists(temp_filepath) and temp_filepath != original_filepath:
            os.remove(temp_filepath)
            
        # Create status message with enabled features
        feature_str = []
        if use_adaptive:
            feature_str.append("Adaptive SNR")
        if use_harmonic:
            feature_str.append("Harmonic Enhancement")
        if use_vocal:
            feature_str.append("Vocal Clarity")
        if use_compression:
            feature_str.append("Dynamic Range Compression")
            
        # Always mention perceptual enhancement (always on)
        feature_str.append("Perceptual Enhancement")
            
        if feature_str:
            feature_message = f"✅ Audio denoised successfully using {', '.join(feature_str)}!"
        else:
            feature_message = "✅ Audio denoised successfully!"
            
        return denoised_filepath, spectrograms_path, iframe_html, feature_message
    
    except Exception as e:
        # Clean up on error
        if temp_filepath and os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except:
                pass
                
        print(f"Error in process_audio: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, f"❌ Error processing audio: {str(e)}"

def generate_metrics_html(metrics_data: Dict[str, Any]) -> str:
    """
    Generate HTML representation of metrics data
    
    Args:
        metrics_data: Dictionary of metrics
        
    Returns:
        str: HTML string
    """
    # Generate HTML for metrics
    html = """
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }
            .metrics-container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .metrics-section {
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background-color: #f9f9f9;
            }
            .metrics-title {
                margin-top: 0;
                color: #333;
                border-bottom: 1px solid #ddd;
                padding-bottom: 8px;
            }
            .metrics-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 8px;
            }
            .metrics-label {
                font-weight: bold;
                color: #555;
            }
            .metrics-value {
                color: #333;
            }
            .improvement-positive {
                color: green;
            }
            .improvement-negative {
                color: red;
            }
            .noise-bar {
                height: 20px;
                background-color: #3498db;
                margin-bottom: 5px;
                border-radius: 3px;
            }
            .noise-label {
                display: flex;
                justify-content: space-between;
            }
            .processing-pills {
                display: flex;
                flex-wrap: wrap;
                gap: 8px;
                margin-top: 10px;
            }
            .processing-pill {
                background-color: #3498db;
                color: white;
                padding: 5px 10px;
                border-radius: 15px;
                font-size: 0.9em;
            }
            .processing-pill.active {
                background-color: #2ecc71;
            }
            .processing-pill.inactive {
                background-color: #95a5a6;
            }
        </style>
    </head>
    <body>
        <div class="metrics-container">
            <div class="metrics-section">
                <h2 class="metrics-title">Audio Enhancement Metrics</h2>
    """
    
    # Add before/after metrics
    html += """
                <h3>Before vs After</h3>
                <div class="metrics-row">
                    <div class="metrics-label">Metric</div>
                    <div class="metrics-label">Before</div>
                    <div class="metrics-label">After</div>
                    <div class="metrics-label">Change</div>
                </div>
    """
    
    # SNR
    snr_before = metrics_data["original"]["estimated_snr"]
    snr_after = metrics_data["processed"]["estimated_snr"]
    snr_change = metrics_data["improvement"]["snr_improvement"]
    change_class = "improvement-positive" if snr_change > 0 else "improvement-negative"
    html += f"""
                <div class="metrics-row">
                    <div class="metrics-label">Estimated SNR (dB)</div>
                    <div class="metrics-value">{snr_before:.2f}</div>
                    <div class="metrics-value">{snr_after:.2f}</div>
                    <div class="metrics-value {change_class}">{snr_change:+.2f}</div>
                </div>
    """
    
    # Spectral Centroid
    sc_before = metrics_data["original"]["spectral_centroid"]
    sc_after = metrics_data["processed"]["spectral_centroid"]
    sc_change = metrics_data["improvement"]["spectral_balance_change"]
    change_class = "improvement-positive" if abs(sc_change) < 500 else "improvement-negative"
    html += f"""
                <div class="metrics-row">
                    <div class="metrics-label">Spectral Centroid</div>
                    <div class="metrics-value">{sc_before:.1f}</div>
                    <div class="metrics-value">{sc_after:.1f}</div>
                    <div class="metrics-value {change_class}">{sc_change:+.1f}</div>
                </div>
    """
    
    # Zero Crossing Rate
    zcr_before = metrics_data["original"]["zero_crossing_rate"]
    zcr_after = metrics_data["processed"]["zero_crossing_rate"]
    zcr_change = -metrics_data["improvement"]["noise_reduction"]  # Note the sign flip for display
    change_class = "improvement-positive" if zcr_change < 0 else "improvement-negative"
    html += f"""
                <div class="metrics-row">
                    <div class="metrics-label">Zero Crossing Rate</div>
                    <div class="metrics-value">{zcr_before:.4f}</div>
                    <div class="metrics-value">{zcr_after:.4f}</div>
                    <div class="metrics-value {change_class}">{zcr_change:+.4f}</div>
                </div>
    """
    
    # Dynamic Range
    dr_before = metrics_data["original"]["dynamic_range_db"]
    dr_after = metrics_data["processed"]["dynamic_range_db"]
    dr_change = dr_after - dr_before
    change_class = "improvement-positive" if dr_change > 0 else "improvement-negative"
    html += f"""
                <div class="metrics-row">
                    <div class="metrics-label">Dynamic Range (dB)</div>
                    <div class="metrics-value">{dr_before:.2f}</div>
                    <div class="metrics-value">{dr_after:.2f}</div>
                    <div class="metrics-value {change_class}">{dr_change:+.2f}</div>
                </div>
    """
    
    # Add noise classification section
    html += """
            </div>
            
            <div class="metrics-section">
                <h2 class="metrics-title">Noise Classification</h2>
                <p>Analysis of detected noise types:</p>
    """
    
    # Add bars for each noise type
    for noise_type, probability in metrics_data["noise_classification"].items():
        width_percent = int(probability * 100)
        display_name = " ".join(word.capitalize() for word in noise_type.split("_"))
        html += f"""
                <div>
                    <div class="noise-bar" style="width: {width_percent}%;"></div>
                    <div class="noise-label">
                        <span>{display_name}</span>
                        <span>{probability:.2f}</span>
                    </div>
                </div>
        """
    
    # Add processing info section
    html += """
            </div>
            
            <div class="metrics-section">
                <h2 class="metrics-title">Processing Techniques Applied</h2>
                <div class="processing-pills">
    """
    
    # Add pills for each processing technique
    processing_info = metrics_data["processing_info"]
    for tech, enabled in processing_info.items():
        status_class = "active" if enabled else "inactive"
        display_name = " ".join(word.capitalize() for word in tech.split("_"))
        html += f"""
                    <div class="processing-pill {status_class}">{display_name}</div>
        """
    
    # Close all divs
    html += """
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

# Define UI colors and theme
primary_color = "#FFD700"  # Gold
secondary_color = "#2C2C2C"  # Dark Gray
background_color = "#1A1A1A"  # Almost Black
text_color = "#FFFFFF"  # White
accent_color = "#FFA500"  # Orange/Yellow

custom_css = f"""
    body, .gradio-container {{
        background-color: {background_color};
        color: {text_color};
    }}
    
    .title-text {{
        color: {primary_color};
        text-align: center;
        margin-bottom: 1rem;
    }}
    
    .footer-text {{
        text-align: center;
        margin-top: 1rem;
        color: {text_color};
    }}
    
    .footer-text a {{
        color: {primary_color};
    }}
    
    .tab-nav * {{
        color: {text_color};
    }}
    
    button.primary {{
        background-color: {primary_color} !important;
        color: {background_color} !important;
    }}
    
    .custom-box {{
        border: 1px solid {primary_color};
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
        background-color: {secondary_color};
    }}
    
    label, .label-wrap {{
        color: {text_color} !important;
    }}
"""

# App title
title = f"""
<h1 class="title-text">Enhanced CleanUNet Audio Denoiser</h1>
<p style="text-align: center;">Remove noise from your audio files using NVIDIA's CleanUNet with novel enhancements</p>
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(
    primary_hue=gr.themes.colors.yellow,
    secondary_hue=gr.themes.colors.gray,
    neutral_hue=gr.themes.colors.gray,
    text_size=gr.themes.sizes.text_md
).set(
    body_background_fill=background_color,
    body_text_color=text_color,
    button_primary_background_fill=primary_color,
    button_primary_text_color=background_color,
    background_fill_primary=secondary_color
)) as demo:
    gr.HTML(title)
    
    with gr.Tabs():
        with gr.TabItem("Upload Audio"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(
                        sources=["upload"],
                        label="Upload Noisy Audio File",
                        elem_id="audio-input",
                        type="filepath"  # Explicitly set type to filepath for uploaded files
                    )
                    
                    # Enhancement options
                    with gr.Group(elem_classes=["custom-box"]):
                        gr.Markdown("### Enhancement Settings")
                        with gr.Row():
                            use_adaptive = gr.Checkbox(
                                label="Adaptive SNR Processing",
                                value=True,
                                info="Dynamically adjusts processing based on noise levels"
                            )
                            use_harmonic = gr.Checkbox(
                                label="Harmonic Enhancement",
                                value=True,
                                info="Enhances speech harmonics for better clarity"
                            )
                        with gr.Row():
                            use_vocal = gr.Checkbox(
                                label="Vocal Clarity",
                                value=True,
                                info="Enhances frequency bands important for speech"
                            )
                            use_compression = gr.Checkbox(
                                label="Dynamic Range Compression",
                                value=False,
                                info="Reduces dynamic range for more consistent volume"
                            )
                        
                        gr.Markdown("*Note: Perceptual Enhancement is always active*")
                    
                    upload_button = gr.Button(
                        "Denoise Audio", 
                        variant="primary",
                        elem_id="denoise-btn"
                    )
                    
                    status_output = gr.Textbox(
                        label="Status", 
                        placeholder="Upload an audio file and click 'Denoise Audio'",
                        interactive=False
                    )
                    
                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("Audio"):
                            audio_output = gr.Audio(
                                label="Denoised Audio", 
                                elem_id="audio-output",
                                interactive=False
                            )
                        with gr.TabItem("Spectrogram"):
                            spectrogram_output = gr.Image(
                                label="Spectrograms (Original vs Denoised)",
                                elem_id="spectrogram-output",
                                interactive=False
                            )
                        with gr.TabItem("Metrics"):
                            metrics_output = gr.HTML(
                                label="Audio Enhancement Metrics",
                                elem_id="metrics-output"
                            )
                    
        with gr.TabItem("Record Audio"):
            with gr.Row():
                with gr.Column():
                    mic_input = gr.Audio(
                        sources=["microphone"],
                        label="Record Audio",
                        elem_id="mic-input"
                    )
                    
                    # Enhancement options for recorded audio
                    with gr.Group(elem_classes=["custom-box"]):
                        gr.Markdown("### Enhancement Settings")
                        with gr.Row():
                            record_adaptive = gr.Checkbox(
                                label="Adaptive SNR Processing",
                                value=True,
                                info="Dynamically adjusts processing based on noise levels"
                            )
                            record_harmonic = gr.Checkbox(
                                label="Harmonic Enhancement",
                                value=True,
                                info="Enhances speech harmonics for better clarity"
                            )
                        with gr.Row():
                            record_vocal = gr.Checkbox(
                                label="Vocal Clarity",
                                value=True,
                                info="Enhances frequency bands important for speech"
                            )
                            record_compression = gr.Checkbox(
                                label="Dynamic Range Compression",
                                value=False,
                                info="Reduces dynamic range for more consistent volume"
                            )
                        
                        gr.Markdown("*Note: Perceptual Enhancement is always active*")
                    
                    record_button = gr.Button(
                        "Denoise Recorded Audio", 
                        variant="primary",
                        elem_id="record-denoise-btn"
                    )
                    
                    record_status = gr.Textbox(
                        label="Status", 
                        placeholder="Record audio and click 'Denoise Recorded Audio'",
                        interactive=False
                    )
                    
                with gr.Column():
                    with gr.Tabs():
                        with gr.TabItem("Audio"):
                            record_output = gr.Audio(
                                label="Denoised Recording", 
                                elem_id="record-output",
                                interactive=False
                            )
                        with gr.TabItem("Spectrogram"):
                            record_spectrogram = gr.Image(
                                label="Spectrograms (Original vs Denoised)",
                                elem_id="record-spectrogram",
                                interactive=False
                            )
                        with gr.TabItem("Metrics"):
                            record_metrics = gr.HTML(
                                label="Audio Enhancement Metrics",
                                elem_id="record-metrics-output"
                            )
        
        with gr.TabItem("About"):
            with gr.Group(elem_classes=["custom-box"]):
                gr.Markdown("""
                ## Enhanced Audio Denoising System
                
                This system enhances NVIDIA's CleanUNet model with several novel approaches:
                
                1. **Adaptive SNR Processing**: Dynamically adjusts denoising parameters based on the signal-to-noise ratio of each audio segment. This optimizes the trade-off between noise reduction and speech preservation.
                
                2. **Perceptual Enhancement**: Applies psychoacoustic principles to enhance the perceived quality of speech, focusing on frequencies most important to human hearing.
                
                3. **Harmonic Enhancement**: Selectively boosts harmonic components of speech, which are critical for intelligibility and naturalness.
                
                4. **Vocal Clarity Enhancement**: Applies targeted processing to frequency bands containing human speech, improving the clarity and presence of vocal content.
                
                5. **Dynamic Range Compression** (optional): Reduces the volume difference between loud and soft parts of the audio, making speech more consistently audible in varying noise environments.
                
                6. **Noise Classification**: Analyzes the characteristics of background noise to optimize processing parameters (shown in metrics).
                
                These enhancements provide significant improvements in audio quality metrics including SNR, spectral balance, and dynamic range compared to the base model, particularly in challenging noise environments.
                """)
    
    # Connect components with processing function
    upload_button.click(
        fn=process_audio,
        inputs=[audio_input, use_adaptive, use_harmonic, use_vocal, use_compression],
        outputs=[audio_output, spectrogram_output, metrics_output, status_output]
    )
    
    record_button.click(
        fn=process_audio,
        inputs=[mic_input, record_adaptive, record_harmonic, record_vocal, record_compression],
        outputs=[record_output, record_spectrogram, record_metrics, record_status]
    )
    
    # Footer information
    footer = """
    <div class="footer-text">
        <p><b>Developed by:</b> Kevin Doshi</p>
        <p><a href="">GitHub Repository</a></p>
    </div>
    """
    gr.HTML(footer)

# Launch the app
if __name__ == "__main__":
    print("Starting Enhanced Gradio app...")
    print("After launching, access the app at: http://127.0.0.1:7862 or http://localhost:7862")
    
    demo.launch(
        server_name="127.0.0.1",  # Changed from 0.0.0.0 to 127.0.0.1 for local access
        server_port=7862,
        share=True,  # Enable sharing for remote access
        inbrowser=True  # Open in browser automatically
    )