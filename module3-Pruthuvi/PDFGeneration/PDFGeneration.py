import hashlib
import os

print("Starting script...")

# --- Monkeypatch hashlib.md5 to ignore 'usedforsecurity' kwarg if not supported ---
def md5_supports_usedforsecurity():
    try:
        hashlib.md5(b'test', usedforsecurity=False)
        return True
    except TypeError:
        return False

if not md5_supports_usedforsecurity():
    print("Monkeypatching hashlib.md5: 'usedforsecurity' not supported on this platform.")
    _md5_orig = hashlib.md5
    def md5_monkeypatched(*args, **kwargs):
        if 'usedforsecurity' in kwargs:
            kwargs.pop('usedforsecurity')
        return _md5_orig(*args, **kwargs)
    hashlib.md5 = md5_monkeypatched
else:
    print("No monkeypatch needed for hashlib.md5.")

# --- Now import the rest of your libraries ---
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage

# --- Register Sinhala font ---
script_dir = os.path.abspath(os.path.dirname(__file__))
SINHALA_FONT_PATH = os.path.join(script_dir, "sinhala_fonts", "Iskoola Pota Regular.ttf")
print(f"Registering Sinhala font from: {SINHALA_FONT_PATH}")
try:
    pdfmetrics.registerFont(TTFont('SinhalaFont', SINHALA_FONT_PATH))
    print("Sinhala font registered successfully.")
except Exception as e:
    print(f"Error registering font: {e}")

# Define your input and output files/folders
root_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
steps_txt = os.path.join(root_folder, 'final_instructions.txt')
images_folder = os.path.join(root_folder, 'BestFrame/best_frames')
output_pdf = os.path.join(root_folder, 'performance_guide.pdf') # output PDF file name

# Read steps and parse index and instruction
print(f"Reading instructions from text file: {steps_txt}")
steps = []
try:
    with open(steps_txt, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, start=1):
            if line.strip():
                parts = line.strip().split('.', 1)
                if len(parts) == 2 and parts[0].strip().isdigit():
                    index = parts[0].strip()
                    instruction = parts[1].strip()
                    steps.append((index, instruction))
                    print(f"Loaded step {index}: {instruction}")
                else:
                    print(f"Skipping line {line_num}: does not match expected 'index.instruction' pattern")
except Exception as e:
    print(f"Error reading steps file: {e}")

print(f"Total steps loaded: {len(steps)}")

# Initialize PDF document
print(f"Initializing PDF document: {output_pdf}")
doc = SimpleDocTemplate(output_pdf, title="Performance guide", pagesize=A4)

# Create stylesheet and add Sinhala style using registered font
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Sinhala',
                          fontName='SinhalaFont',
                          fontSize=12,
                          leading=15))

story = []
print("Adding title to the PDF.")
story.append(Paragraph("Performance guide", styles['Title']))
story.append(Spacer(1, 0.25 * inch))

# Process each step with optional image
for index, instruction in steps:
    print(f"Adding step {index} to PDF.")
    # Add instruction with Sinhala font style
    story.append(Paragraph(f"{index}. {instruction}", styles['Sinhala']))
    story.append(Spacer(1, 0.1 * inch))

    # Search for relevant image by index with common image extensions
    found_image = None
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'PNG']:
        img_path = os.path.join(images_folder, f"{index}.{ext}")
        if os.path.isfile(img_path):
            found_image = img_path
            print(f"Found image for step {index} at: {found_image}")
            break

    # If found, resize to max width and add image after instruction
    if found_image:
        try:
            img = PILImage.open(found_image)
            width, height = img.size
            max_width = 5.5 * inch  # max image width in PDF
            if width > max_width:
                ratio = max_width / float(width)
                width = max_width
                height = float(height) * ratio
            story.append(Image(found_image, width, height))
            story.append(Spacer(1, 0.2 * inch))
            print(f"Added image for step {index} (resized to width {width} pts).")
        except Exception as e:
            print(f"Error adding image for step {index}: {e}")
    else:
        print(f"No image found for step {index}.")

print("Building PDF document...")
try:
    doc.build(story)
    print(f"Generated PDF successfully: {output_pdf}")
except Exception as e:
    print(f"Error while building PDF: {e}")
