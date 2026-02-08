import os
import requests
import fitz  # PyMuPDF
from tqdm import tqdm
from requests.exceptions import RequestException

def download_and_convert_pdfs(urls, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    errors = []

    for idx, url in enumerate(tqdm(urls, desc="Downloading and Converting PDFs")):
        try:
            # Download the PDF
            response = requests.get(url)
            response.raise_for_status()  # Check for HTTP errors

            # Create a temporary file to save the PDF
            temp_pdf_filename = f"temp_document_{idx + 1}.pdf"
            with open(temp_pdf_filename, 'wb') as f:
                f.write(response.content)

            # Attempt to open the PDF using PyMuPDF
            pdf_document = fitz.open(temp_pdf_filename)

            # Create a unique directory for each PDF
            pdf_folder = os.path.join(save_folder, f"document_{idx + 1}")
            if not os.path.exists(pdf_folder):
                os.makedirs(pdf_folder)

            # Convert PDF to PNG using PyMuPDF
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)  # Load page
                pix = page.get_pixmap()  # Render page to an image
                image_filename = os.path.join(pdf_folder, f"document_{idx + 1}_page_{page_num + 1}.png")
                pix.save(image_filename)

            # Close the PDF document
            pdf_document.close()

            # Delete the temporary PDF file
            os.remove(temp_pdf_filename)

        except RequestException:
            errors.append(f"Failed to download: {url}")
        except Exception as e:
            errors.append(f"An error occurred with document_{idx + 1}: {e}")
            # Clean up if there's an error during conversion
            if os.path.exists(temp_pdf_filename):
                os.remove(temp_pdf_filename)
            if os.path.exists(pdf_folder):
                for file in os.listdir(pdf_folder):
                    os.remove(os.path.join(pdf_folder, file))
                os.rmdir(pdf_folder)

    # Save errors to a file
    if errors:
        parent_directory = os.path.dirname(save_folder)
        error_log_filename = os.path.join(parent_directory, "error_log.txt")
        with open(error_log_filename, 'w') as error_log:
            for error in errors:
                error_log.write(error + "\n")

def read_urls_from_file(file_path):
    with open(file_path, 'r') as file:
        urls = file.read().strip().split('\n')
    return urls

# Usage:
file_path = '/home/shodh/onnx/pdf_links.txt'  # Path to the text file containing URLs
urls = read_urls_from_file(file_path)

save_folder = 'Url_PDF_Img'  # Name of the save folder
save_path = os.path.join(os.getcwd(), save_folder)  # Creating the full path using current working directory

# Check if the folder already exists, if not, create it
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print(f"Save folder '{save_folder}' created successfully at: {save_path}")
else:
    print(f"Save folder '{save_folder}' already exists at: {save_path}")

download_and_convert_pdfs(urls, save_folder)

import os
import onnxruntime_genai as og

prompt = ''' <|user|>

<|image_1|>:
If you identify any diagram in the page provided, describe the diagram
<|end|>
<|assistant|> 
'''

# Initialize the model
model = og.Model('cuda-int4-rtn-block-32')
processor = model.create_multimodal_processor()
tokenizer_stream = processor.create_stream()

# Define the main directory containing subfolders with images
main_dir = save_folder

# List all subfolders in the main directory
subfolders = [f.path for f in os.scandir(main_dir) if f.is_dir()]

# Define the output folder and the data file name
output_folder = "Output_Image_Caption"
output_file_name = "data.txt"

# Create the full path using the current working directory
output_folder_path = os.path.join(os.getcwd(), output_folder)
output_file_path = os.path.join(output_folder_path, output_file_name)

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

# Iterate over each subfolder
for subfolder in subfolders:
    # List all image files in the subfolder
    image_files = [f for f in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, f))]
    
    # Process each image in the subfolder
    for image_file in image_files:
        # Open the image
        image_path = os.path.join(subfolder, image_file)
        image = og.Images.open(image_path)

        # Process the image with the same prompt and inputs
        inputs = processor(prompt, images=image)
        params = og.GeneratorParams(model)
        params.set_inputs(inputs)
        params.set_search_options(max_length=3000)
        generator = og.Generator(model, params)
        
        response = ""

        # Generate response
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()

            new_token = generator.get_next_tokens()[0]
            response += tokenizer_stream.decode(new_token)

        # Save the response to the text file
        with open(output_file_path, 'a') as file:
            file.write(f"Image: {image_path}\n")
            file.write(response)
            file.write("#$%---#$%")  # Separate responses for different images
        
        # Clean up
        del generator