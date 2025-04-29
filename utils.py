import streamlit as st
import openai
import os
import io
import json
import random
import pdfplumber
import docx
import re
import pandas as pd
from docx import Document

# Set up OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["openai"]["api_key"]
openai.api_key = openai_api_key

# GenAI helper class
class GenAI:
    def __init__(self, openai_api_key):
        self.client = openai.Client(api_key=openai_api_key)
        self.openai_api_key = openai_api_key

    def generate_text(self, prompt, instructions="You are a helpful AI assistant", model="gpt-4o-mini", response_format="text", temperature=1.0):
        if response_format == "json":
            response_format_obj = {"type": "json_object"}
        else:
            response_format_obj = {"type": response_format}

        completion = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format=response_format_obj,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt}
            ]
        )
        return completion.choices[0].message.content

    def generate_chat_response(self, chat_history, user_message, instructions, model="gpt-4o-mini", output_type="text"):
        chat_history.append({"role": "user", "content": user_message})
        completion = self.client.chat.completions.create(
            model=model,
            response_format={"type": output_type},
            messages=[
                {"role": "system", "content": instructions},
                *chat_history
            ]
        )
        response = completion.choices[0].message.content
        chat_history.append({"role": "assistant", "content": response})
        return response

# Resume parsing functions
def extract_text_from_file(file):
    file_type = file.name.split(".")[-1].lower()
    file_content = file.read()
    if file_type == "pdf":
        return extract_text_from_pdf(file_content)
    elif file_type == "docx":
        return extract_text_from_docx(file_content)
    else:
        return "Unsupported file format"

def extract_text_from_pdf(file_content):
    try:
        pdf_file = io.BytesIO(file_content)
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except Exception as e:
        print(f"PDF Extraction Failed: {e}")
        return "Failed to extract text."

def extract_text_from_docx(file_content):
    try:
        docx_file = io.BytesIO(file_content)
        doc = docx.Document(docx_file)
        return "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"DOCX Extraction Failed: {e}")
        return "Failed to extract text."

def parse_resume(file):
    content = extract_text_from_file(file)
    gen_ai = GenAI(openai_api_key)
    prompt = f"""
    Parse the following resume text into a structured JSON format with the following sections:
    - summary
    - experience (title, company, start_date, end_date, description)
    - education (degree, institution, start_date, end_date, gpa)
    - skills
    - contact (name, email, phone, location)

    Resume text:
    {content}

    Return only the JSON object.
    """
    try:
        response = gen_ai.generate_text(
            prompt=prompt,
            instructions="You are an expert resume parser. Extract structured data from the resume accurately.",
            model="gpt-4o-mini",
            response_format="json",
            temperature=0.2
        )
        parsed = json.loads(response) if isinstance(response, str) else response
        parsed["raw_text"] = content
        return parsed
    except Exception as e:
        print(f"GPT parsing failed: {e}")
        return {"raw_text": content, "summary": "", "experience": [], "education": [], "skills": [], "contact": {}}

def get_resume_tips(parsed_resume):
    gen_ai = GenAI(openai_api_key)
    resume_json = json.dumps(parsed_resume, indent=2)
    prompt = f"""
    You must return a JSON array of 5 to 7 tips. Each tip must be a string.

    Now, based on this parsed resume:

    {resume_json}

    Generate 5-7 resume improvement tips following the format exactly.
    """
    try:
        response = gen_ai.generate_text(
            prompt=prompt,
            instructions="You are an expert resume coach. Provide helpful, clear, and specific advice.",
            model="gpt-4o-mini",
            response_format="json",
            temperature=0.7
        )
        tips = json.loads(response) if isinstance(response, str) else response
        if not (isinstance(tips, list) and all(isinstance(tip, str) for tip in tips)):
            raise ValueError("Tips must be a list of strings")
        return tips
    except Exception as e:
        print(f"Resume tips fallback triggered: {e}")
        return [
            "Add specific numbers to your achievements (e.g., increased sales by 20%).",
            "Tailor your resume to the job you're applying for by using relevant keywords.",
            "Highlight leadership or team roles you have held.",
            "Keep formatting clean and consistent.",
            "Place the most relevant skills and experiences at the top of your resume."
        ]

def check_openai_key():
    try:
        openai.Model.list()
        return True
    except Exception:
        return False