import os
import pdfplumber

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Ensure there's text
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None
    return text


def process_pdfs(input_dir, output_dir):
    """Read PDF files from input directory and save extracted text to output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            
            output_file = os.path.join(output_dir, filename.replace(".pdf", ".txt"))
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(text)
            
            print(f"Processed: {filename} -> {output_file}")

if __name__ == "__main__":
    input_directory = ""  # Change this to your input directory with PDF files
    output_directory = ""  # Change this to your output directory
    
    process_pdfs(input_directory, output_directory)
