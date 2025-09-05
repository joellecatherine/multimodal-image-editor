# Multimodal Context-Aware Image Editing

This project is a functional prototype of a multimodal image editing tool that uses a text prompt and a user-defined region of interest (ROI) to perform context-aware inpainting with a diffusion model (Stable Diffusion).

## Key Features

-   **Interactive UI**: A simple web interface built with Streamlit for uploading images.
-   **Region of Interest (ROI) Selection**: Users can define a precise rectangular area for editing.
-   **Text-Guided Inpainting**: Leverages a pre-trained Stable Diffusion model to fill the selected region based on a natural language prompt.
-   **Multimodal Input**: Combines image data (the source and mask) and text data (the prompt) to guide the generative process.
-   **Efficient Implementation**: Uses a model from the Hugging Face ecosystem (`diffusers`) and caches resources to ensure a responsive user experience.

## Demo

![Application Demo](demo_img/demo_img.jpeg)

## Tech Stack

-   **Backend**: Python 3.11+
-   **Dependency Management**: Poetry
-   **Deep Learning**: PyTorch
-   **AI/ML Framework**: Hugging Face (`diffusers`, `transformers`)
-   **Web UI**: Streamlit
-   **Image Processing**: Pillow, NumPy

## Setup and Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management to ensure reproducible builds.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/joellecatherine/multimodal-image-editor.git](https://github.com/joellecatherine/multimodal-image-editor.git)
    cd multimodal-image-editor
    ```

2.  **Install Poetry:**
    Follow the official instructions at [python-poetry.org](https://python-poetry.org/docs/#installation).

3.  **Install dependencies:**
    This command creates a virtual environment using Python 3.11+ and installs all required libraries from the `poetry.lock` file.
    ```bash
    poetry install
    ```

## Usage

Launch the Streamlit application from your terminal:

```bash
poetry run streamlit run app.py