import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import os
from transformers import pipeline
import textwrap

# --- 1. SETUP AND MODEL LOADING ---

# Set the page configuration for the Streamlit app
st.set_page_config(layout="wide", page_title="AI Meme & Poster Creator")


# Cache the model loading to prevent reloading on every interaction
@st.cache_resource
def load_model():
    """Loads the text generation model from Hugging Face."""
    return pipeline('text-generation', model='distilgpt2')


# Load the AI model for text generation
try:
    text_generator = load_model()
except Exception as e:
    st.error(f"Failed to load the model. Please check your internet connection. Error: {e}")
    st.stop()

# Define the path to templates and the font file
TEMPLATE_DIR = "templates"
# Use the more comprehensive NotoSans font to support more characters
FONT_FILE = "NotoSans-VariableFont_wdth,wght.ttf"


# --- 2. CORE FUNCTIONS ---

def generate_caption(prompt_text):
    """Generates a caption using the AI model."""
    if not prompt_text:
        return "Please enter a topic!"

    # We add a bit of instruction to the prompt to guide the AI
    full_prompt = f"A short, funny meme caption for an image, on the topic of {prompt_text}:"

    try:
        # Generate text using the model
        sequences = text_generator(
            full_prompt,
            max_length=50,  # Gives the AI more space to be creative
            num_return_sequences=1,
            no_repeat_ngram_size=2,  # Prevents the AI from repeating phrases
            num_beams=5,  # Generates higher-quality text
            early_stopping=True
        )
        # Clean up the generated text to get just the caption
        generated_text = sequences[0]['generated_text'].replace(full_prompt, "").strip()
        return generated_text if generated_text else "AI could not generate a caption."
    except Exception as e:
        st.error(f"Error during AI caption generation: {e}")
        return "Error generating text."


def create_image_with_text(template_path, top_text):
    """Overlays text onto an image template."""
    try:
        # Open the base image
        img = Image.open(template_path)
        draw = ImageDraw.Draw(img)

        # Define image size and font size
        image_width, image_height = img.size
        font_size = int(image_width / 18)  # Adjusted font size for better fit

        # Load the font
        try:
            font = ImageFont.truetype(FONT_FILE, font_size)
        except IOError:
            st.error(f"Font file '{FONT_FILE}' not found. Please ensure it's in your project folder.")
            return None

        # Wrap text to fit the image width
        avg_char_width = sum(font.getbbox(char)[2] for char in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ') / 26
        chars_per_line = int(image_width * 0.9 / avg_char_width)
        wrapped_text = textwrap.fill(top_text.upper(), width=chars_per_line)

        # Calculate text position (center top)
        text_bbox = draw.textbbox((0, 0), wrapped_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        position_x = (image_width - text_width) / 2
        position_y = 15  # A small padding from the top

        # Draw text with a black outline for better readability
        outline_color = "black"
        for dx, dy in [(-2, -2), (-2, 2), (2, -2), (2, 2)]:
            draw.text((position_x + dx, position_y + dy), wrapped_text, font=font, fill=outline_color, align="center")

        # Draw the main white text
        draw.text((position_x, position_y), wrapped_text, font=font, fill="white", align="center")

        return img
    except FileNotFoundError:
        st.error(f"Template image not found at {template_path}")
        return None
    except Exception as e:
        st.error(f"An error occurred during image creation: {e}")
        return None


# --- 3. STREAMLIT UI ---

st.title("🚀 AI Meme & Poster Creator")
st.markdown("Create fun content in seconds. Just choose a template, enter a topic, and let the AI do the work!")

# Get the list of available templates
try:
    templates = [f for f in os.listdir(TEMPLATE_DIR) if f.endswith(('png', 'jpg', 'jpeg'))]
    if not templates:
        st.error(f"No templates found in the '{TEMPLATE_DIR}' folder. Please add some images.")
        st.stop()
except FileNotFoundError:
    st.error(f"The '{TEMPLATE_DIR}' folder was not found. Please create it and add image templates.")
    st.stop()

# UI components in the sidebar
with st.sidebar:
    st.header("🎨 Controls")
    selected_template = st.selectbox("1. Choose a Template", templates)
    prompt = st.text_input("2. Enter a Topic for the Meme", placeholder="e.g., studying for exams")
    generate_button = st.button("3. Generate Meme!", type="primary")

# Main content area
col1, col2 = st.columns(2)

# Display the original template
with col1:
    st.subheader("Template")
    template_path = os.path.join(TEMPLATE_DIR, selected_template)
    st.image(template_path, use_container_width=True)  # Updated to use_container_width

# Generate and display the final meme
with col2:
    st.subheader("Your AI-Generated Meme")
    if generate_button:
        with st.spinner("AI is thinking... ✨"):
            # Generate the caption
            caption = generate_caption(prompt)

            # Create the final image
            if caption and "Error" not in caption:
                final_image = create_image_with_text(template_path, caption)

                if final_image:
                    st.image(final_image, use_container_width=True)  # Updated to use_container_width
                    # Convert image to bytes for download
                    final_image.save("meme_output.png")
                    with open("meme_output.png", "rb") as file:
                        st.download_button(
                            label="Download Meme",
                            data=file,
                            file_name="ai_meme.png",
                            mime="image/png"
                        )
                else:
                    st.warning("Could not generate the final image.")
            else:
                st.error(caption)  # Display error if caption generation failed
    else:
        st.info("Click the 'Generate Meme!' button to create your masterpiece.")