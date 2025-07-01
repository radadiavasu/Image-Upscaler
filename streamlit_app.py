import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
import io
import base64
from datetime import datetime
import tempfile
import zipfile

# Import Real-ESRGAN components
try:
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    REALESRGAN_AVAILABLE = True
except ImportError:
    REALESRGAN_AVAILABLE = False
    st.error("Real-ESRGAN not installed. Please install with: pip install realesrgan")

# Page configuration
st.set_page_config(
    page_title="Real-ESRGAN Image Upscaler",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.25rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def get_model_paths():
    """Get model paths, checking both local and original directories"""
    try:
        # Try local directory first
        model_dir = os.path.dirname(os.path.abspath(__file__))
        local_paths = {
            "RealESRGAN_x4plus": os.path.join(model_dir, "weights", "RealESRGAN_x4plus.pth"),
            "RealESRGAN_x4plus_anime": os.path.join(model_dir, "weights", "RealESRGAN_x4plus_anime_6B.pth")
        }
        
        # Check if local files exist
        for model_name, path in local_paths.items():
            if os.path.exists(path):
                return local_paths
        
        # Fallback to original path structure
        original_paths = {
            "RealESRGAN_x4plus": r'G:\real-esrgan\weights\RealESRGAN_x4plus.pth',
            "RealESRGAN_x4plus_anime": r'G:\real-esrgan\weights\RealESRGAN_x4plus_anime_6B.pth'
        }
        
        return original_paths
    except:
        # Final fallback
        return {
            "RealESRGAN_x4plus": "weights/RealESRGAN_x4plus.pth",
            "RealESRGAN_x4plus_anime": "weights/RealESRGAN_x4plus_anime_6B.pth"
        }

@st.cache_resource
def load_model(model_name, scale, tile_size, tile_pad, use_gpu, half_precision):
    """Load and cache the Real-ESRGAN model"""
    if not REALESRGAN_AVAILABLE:
        raise ImportError("Real-ESRGAN not available")
    
    # Set device
    device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
    
    # Get model paths
    model_paths = get_model_paths()
    model_path = model_paths.get(model_name)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load RRDBNet model architecture
    if "anime" in model_name:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=6,
            num_grow_ch=32, scale=scale
        )
    else:
        model = RRDBNet(
            num_in_ch=3, num_out_ch=3,
            num_feat=64, num_block=23,
            num_grow_ch=32, scale=scale
        )
    
    # Load Real-ESRGANer
    upsampler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model,
        tile=tile_size if tile_size > 0 else None,
        tile_pad=tile_pad,
        pre_pad=0,
        half=half_precision and torch.cuda.is_available(),
        device=device
    )
    
    return upsampler, device

def process_image(upsampler, input_image, progress_bar=None):
    """Process image with Real-ESRGAN"""
    # Convert PIL Image to numpy array
    img_np = np.array(input_image)
    
    # Update progress
    if progress_bar:
        progress_bar.progress(10)
    
    # Enhance the image
    output, _ = upsampler.enhance(img_np)
    
    # Update progress
    if progress_bar:
        progress_bar.progress(90)
    
    # Convert back to PIL Image
    output_image = Image.fromarray(output)
    
    # Complete progress
    if progress_bar:
        progress_bar.progress(100)
    
    return output_image

def get_download_link(img, filename):
    """Generate download link for image"""
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    b64 = base64.b64encode(buffer.read()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">📥 Download {filename}</a>'

def main():
    # Header
    st.markdown('<h1 class="main-header">🖼️ Real-ESRGAN Image Upscaler</h1>', unsafe_allow_html=True)
    
    if not REALESRGAN_AVAILABLE:
        st.error("Real-ESRGAN is not installed. Please install the required dependencies.")
        st.code("pip install realesrgan basicsr")
        return
    
    # Sidebar for controls
    st.sidebar.title("⚙️ Settings")
    
    # Model settings
    st.sidebar.subheader("Model Configuration")
    
    model_name = st.sidebar.selectbox(
        "Model",
        ["RealESRGAN_x4plus", "RealESRGAN_x4plus_anime"],
        help="Choose the appropriate model for your image type"
    )
    
    scale = st.sidebar.selectbox(
        "Scale Factor",
        [2, 4],
        index=1,
        help="How much to upscale the image"
    )
    
    tile_size = st.sidebar.slider(
        "Tile Size",
        min_value=0,
        max_value=512,
        value=256,
        step=32,
        help="Tile size for processing large images (0 = no tiling)"
    )
    
    tile_pad = st.sidebar.slider(
        "Tile Padding",
        min_value=0,
        max_value=100,
        value=10,
        help="Padding for tiles to avoid artifacts"
    )
    
    # Hardware settings
    st.sidebar.subheader("Hardware Settings")
    
    use_gpu = st.sidebar.checkbox(
        "Use GPU",
        value=torch.cuda.is_available(),
        disabled=not torch.cuda.is_available(),
        help="Use GPU acceleration if available"
    )
    
    half_precision = st.sidebar.checkbox(
        "Half Precision",
        value=False,
        disabled=not torch.cuda.is_available(),
        help="Use half precision for faster processing (GPU only)"
    )
    
    # Device info
    device_info = "🖥️ GPU Available" if torch.cuda.is_available() else "💻 CPU Only"
    st.sidebar.info(f"Device: {device_info}")
    
    # Main content area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Input Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff', 'webp'],
            help="Upload an image to upscale"
        )
        
        if uploaded_file is not None:
            # Load and display input image
            input_image = Image.open(uploaded_file).convert('RGB')
            st.image(input_image, caption=f"Input: {uploaded_file.name}", use_column_width=True)
            
            # Image info
            width, height = input_image.size
            file_size = len(uploaded_file.getvalue()) / 1024  # KB
            st.info(f"📏 Dimensions: {width}×{height}\n📦 Size: {file_size:.1f} KB")
            
            # Recommend tile size
            if tile_size == 0:
                recommended_tile = min(512, max(128, min(width, height) // 4))
                if width * height > 1000000:  # Large image
                    st.warning(f"💡 Large image detected. Consider using tile size: {recommended_tile}")
        else:
            st.info("👆 Upload an image to get started")
    
    with col2:
        st.subheader("📥 Output Image")
        
        if uploaded_file is not None:
            # Process button
            if st.button("🚀 Process Image", type="primary", use_container_width=True):
                try:
                    with st.spinner("Loading model..."):
                        # Load model
                        upsampler, device = load_model(
                            model_name, scale, tile_size, tile_pad, use_gpu, half_precision
                        )
                    
                    # Create progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    status_text.text("Processing image...")
                    
                    # Process image
                    output_image = process_image(upsampler, input_image, progress_bar)
                    
                    # Store in session state
                    st.session_state.output_image = output_image
                    st.session_state.input_filename = uploaded_file.name
                    
                    status_text.text("✅ Processing complete!")
                    
                except FileNotFoundError as e:
                    st.error(f"❌ Model file not found: {str(e)}")
                    st.info("📋 Please ensure model files are in the correct location:")
                    model_paths = get_model_paths()
                    for name, path in model_paths.items():
                        st.code(f"{name}: {path}")
                
                except Exception as e:
                    st.error(f"❌ Error processing image: {str(e)}")
        
        # Display output image if available
        if hasattr(st.session_state, 'output_image') and st.session_state.output_image is not None:
            st.image(
                st.session_state.output_image,
                caption="Output Image",
                use_column_width=True
            )
            
            # Output image info
            out_width, out_height = st.session_state.output_image.size
            st.success(f"📏 Output Dimensions: {out_width}×{out_height}")
            
            # Download button
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name, ext = os.path.splitext(st.session_state.input_filename)
            download_filename = f"{name}_upscaled_{timestamp}.png"
            
            # Convert image to bytes for download
            buffer = io.BytesIO()
            st.session_state.output_image.save(buffer, format='PNG')
            buffer.seek(0)
            
            st.download_button(
                label="📥 Download Result",
                data=buffer.getvalue(),
                file_name=download_filename,
                mime="image/png",
                type="secondary",
                use_container_width=True
            )
        else:
            st.info("🔄 Process an image to see results here")
    
    # Additional information
    st.markdown("---")
    
    with st.expander("ℹ️ About Real-ESRGAN"):
        st.markdown("""
        **Real-ESRGAN** is a practical algorithm for general image/video restoration based on ESRGAN.
        
        **Model Information:**
        - **RealESRGAN_x4plus**: General purpose model for natural images
        - **RealESRGAN_x4plus_anime**: Specialized model for anime/cartoon images
        
        **Tips for best results:**
        - Use appropriate model for your image type
        - For large images, use tiling to avoid memory issues
        - GPU acceleration significantly speeds up processing
        - Half precision can provide faster processing on compatible GPUs
        """)
    
    with st.expander("🔧 Deployment Instructions"):
        st.markdown("""
        **To deploy this app for network access:**
        
        1. **Install dependencies:**
        ```bash
        pip install streamlit realesrgan basicsr
        ```
        
        2. **Run on network:**
        ```bash
        streamlit run app.py --server.address 0.0.0.0 --server.port 8501
        ```
        
        3. **Access from other PCs:**
        - Find your PC's IP address: `ipconfig` (Windows) or `ifconfig` (Linux/Mac)
        - Access via: `http://YOUR_IP_ADDRESS:8501`
        
        4. **Firewall considerations:**
        - Ensure port 8501 is open in your firewall
        - Windows: Allow Streamlit through Windows Defender Firewall
        
        **Security Note:** Only run on trusted networks. For production use, consider authentication and HTTPS.
        """)

if __name__ == "__main__":
    main()
