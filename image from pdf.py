import streamlit as st
import fitz  # PyMuPDF
from PIL import Image
import io

# Function to extract images from PDF
def extract_images_from_pdf(pdf_file):
    # Open the PDF file
    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    images = []

    # Iterate through each page
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))

            # Store image
            images.append(image)

    return images

# Streamlit app
def main():
    st.title('PDF Image Extractor')

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        st.write("Processing the PDF...")
        # Extract images from the uploaded PDF
        images = extract_images_from_pdf(uploaded_file)

        if images:
            st.write(f"Found {len(images)} image(s) in the PDF.")
            for idx, image in enumerate(images):
                st.image(image, caption=f"Image {idx + 1}", use_column_width=True)
        else:
            st.write("No images found in the PDF.")

if __name__ == '__main__':
    main()
