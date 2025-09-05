import streamlit as st
import torch
from diffusers import AutoPipelineForInpainting
from PIL import Image, ImageDraw
import numpy as np

st.set_page_config(layout="wide", page_title="Multimodal Context-Aware Image Editing")

@st.cache_resource
def load_model():
    """Loads the model and configures it for M2/M-series chips."""
    model_path = "stable-diffusion-v1-5/stable-diffusion-inpainting"
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(model_path, torch_dtype=torch.float16, variant="fp16")

        # --- M2 Chip Optimization ---
        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu" # Fallback

        pipe = pipe.to(device)
        return pipe, device
    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        return None, None

def generate_mask(image_size, coordinates):
    """Creates a binary mask image from user-defined coordinates."""
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle(coordinates, fill=255)
    return mask

st.title("Multimodal Context-Aware Image Editor")
st.markdown("""
This app demonstrates image inpainting using a diffusion model. 
Upload an image, define a region to edit, and provide a text prompt to guide the transformation.
""")

pipe, device = load_model()

if pipe is None:
    st.stop()

# --- UI Elements ---
st.sidebar.header("Controls")
st.sidebar.markdown(f"**Model running on: `{device.upper()}`**") # Display the device being used
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
prompt = st.sidebar.text_area("Enter your editing prompt:", "A bright red lighthouse.")
generate_button = st.sidebar.button("Generate Image", type="primary")

col1, col2, col3 = st.columns(3)

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    img_width, img_height = original_image.size

    with col1:
        st.subheader("1. Original Image")
        st.image(original_image, use_container_width=True)

        st.sidebar.subheader("Define Region of Interest (ROI)")

        # --- Set default start coordinates, ensuring they are within the image bounds ---
        # Use 100, but if the image width is smaller, use a value that fits.
        default_x_start = min(100, img_width - 10 if img_width > 10 else 0)
        # Use 250, but if the image height is smaller, use a value that fits.
        default_y_start = min(250, img_height - 10 if img_height > 10 else 0)

        # --- Define a default size for the ROI box ---
        default_roi_width = img_width // 4
        default_roi_height = img_height // 4
        
        # --- Create the sliders using the new default values ---
        x_min = st.sidebar.slider("ROI X-start", 0, img_width, default_x_start)
        y_min = st.sidebar.slider("ROI Y-start", 0, img_height, default_y_start)
        roi_width = st.sidebar.slider("ROI Width", 10, img_width - x_min, default_roi_width)
        roi_height = st.sidebar.slider("ROI Height", 10, img_height - y_min, default_roi_height)
        

    mask_coords = (x_min, y_min, x_min + roi_width, y_min + roi_height)
    mask_image = generate_mask(original_image.size, mask_coords)

    with col2:
        st.subheader("2. Generated Mask")
        st.image(mask_image, use_container_width=True, caption="White area will be edited.")

    if generate_button:
        if not prompt:
            st.warning("Please enter a text prompt.")
        else:
            with st.spinner(f"Generating image on {device.upper()}..."):
                try:
                    input_image_resized = original_image.resize((512, 512))
                    mask_image_resized = mask_image.resize((512, 512))

                    # Generate the 512x512 edited image from the model
                    generated_image_512 = pipe(
                        prompt=prompt,
                        image=input_image_resized,
                        mask_image=mask_image_resized
                    ).images[0]
                    
                    # Upscale the small generated image to the original's dimensions
                    upscaled_generated_image = generated_image_512.resize(original_image.size)
                    
                    # Combine the upscaled result with the original image using the original mask.
                    # This ensures only the edited region is blended in at full resolution.
                    final_image = Image.composite(upscaled_generated_image, original_image, mask_image)
                    
                    with col3:
                        st.subheader("3. Final Image")
                        st.image(final_image, use_container_width=True, caption=f"Prompt: '{prompt}'")
                except Exception as e:
                    st.error(f"An error occurred during image generation: {e}")
else:
    st.info("Please upload an image to begin.")