import os
import io
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
from xhtml2pdf import pisa
import pypdf
from docx import Document

app = FastAPI()

# 1. CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Setup Google AI
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# Use the stable Flash model
model = genai.GenerativeModel('gemini-1.5-flash')

def extract_text_from_file(file_bytes, filename):
    """Helper to pull text out of PDF, DOCX, or TXT"""
    text = ""
    try:
        if filename.endswith('.pdf'):
            pdf_reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
                
        elif filename.endswith('.docx'):
            doc = Document(io.BytesIO(file_bytes))
            for para in doc.paragraphs:
                text += para.text + "\n"
                
        elif filename.endswith('.txt'):
            text = file_bytes.decode('utf-8')
            
    except Exception as e:
        return f"[Error reading file {filename}: {str(e)}]"
    
    return text

async def stream_gemini(prompt_text, image_parts):
    try:
        # Combine text and images for the AI
        input_data = [prompt_text]
        if image_parts:
            input_data.extend(image_parts)

        response = model.generate_content(input_data, stream=True)
        for chunk in response:
            if chunk.text:
                clean_text = chunk.text.replace("\n", "\\n")
                yield f"data: {clean_text}\n\n"
    except Exception as e:
        yield f"data: [ERROR] {str(e)}\n\n"

@app.post("/chat")
async def chat_endpoint(
    prompt: str = Form(...),
    files: list[UploadFile] = File(None)
):
    image_parts = []
    file_texts = ""

    if files:
        for file in files:
            content = await file.read()
            filename = file.filename.lower()

            # If it's an image, send to AI Vision
            if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_parts.append({
                    "mime_type": file.content_type,
                    "data": content
                })
            # If it's a document, extract text
            elif filename.endswith(('.pdf', '.docx', '.txt')):
                extracted = extract_text_from_file(content, filename)
                file_texts += f"\n--- Content of {filename} ---\n{extracted}\n"

    # Create the final prompt
    system_instruction = """
    You are an expert document writer. 
    1. Output your answer using HTML tags only (<h1>, <p>, <b>, <ul>, etc.).
    2. Do NOT use markdown.
    3. Use the user's uploaded file content to write the document.
    """
    
    full_prompt = f"{system_instruction}\n\nUser Prompt: {prompt}\n\nUploaded Document Content:{file_texts}"

    return StreamingResponse(stream_gemini(full_prompt, image_parts), media_type="text/event-stream")

@app.post("/generate-pdf")
async def generate_pdf(html_content: str = Form(...)):
    # Simple PDF Styler
    full_html = f"""
    <html>
    <head>
        <style>
            @page {{ size: A4; margin: 2cm; }}
            body {{ font-family: Helvetica; line-height: 1.5; color: #333; }}
            h1 {{ color: #2563eb; border-bottom: 2px solid #eee; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>{html_content}</body>
    </html>
    """
    pdf_buffer = io.BytesIO()
    pisa.CreatePDF(io.StringIO(full_html), dest=pdf_buffer)
    pdf_buffer.seek(0)
    return Response(content=pdf_buffer.read(), media_type="application/pdf")
