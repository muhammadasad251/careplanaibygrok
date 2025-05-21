from fastapi import FastAPI, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os
from PyPDF2 import PdfReader
import io
import re
import logging
import json
import spacy
from datetime import datetime
from requests.exceptions import ConnectionError, HTTPError, ReadTimeout
from time import sleep
import os
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
nlp = spacy.load("en_core_web_sm")      # Load spaCy English model
def extract_data_from_pdf(text: str) -> dict:
    try:
        logger.debug("Extracting data from PDF text...")
        data = {
            "patient_name": "Unknown",
            "date_of_birth": "Unknown",
            "gender": "Unknown",
            "nhs_number": "Unknown",
            "date_of_assessment": "Unknown",
            "medical_history": "No medical history provided",
            "mobility": "Unknown"
        }

        doc = nlp(text)
        for ent in doc.ents:
            if ent.label_ == "PERSON" and data["patient_name"] == "Unknown":
                data["patient_name"] = ent.text
            elif ent.label_ == "DATE":
                try:
                    parsed_date = datetime.strptime(ent.text, "%d/%m/%Y")
                    if parsed_date.year < 2000:
                        if data["date_of_birth"] == "Unknown":
                            data["date_of_birth"] = ent.text
                    else:
                        if data["date_of_assessment"] == "Unknown":
                            data["date_of_assessment"] = ent.text
                except ValueError:
                    continue

        patterns = {
            "patient_name": r"Name[:\s]*(.*?)(?:\n|$)",
            "date_of_birth": r"(?:Date of Birth|DOB)[:\s]*(.*?)(?:\n|$)",
            "gender": r"Gender[:\s]*(Male|Female|Other|[^\n]*)?(?:\n|$)",
            "nhs_number": r"(?:NHS Number|NHS No\.?)[:\s]*(\d{3}\s*\d{3}\s*\d{4}|[^\n]*)?(?:\n|$)",
            "date_of_assessment": r"(?:Date of Assessment|Assessment Date)[:\s]*(.*?)(?:\n|$)",
            "medical_history": r"(?:Medical History|History)[:\s]*(.*?)(?=\n\n|\Z)",
            "mobility": r"Mobility[:\s]*(.*?)(?:\n|$)"
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match and match.group(1).strip():
                data[key] = match.group(1).strip()

        if data["medical_history"] != "No medical history provided":
            data["medical_history"] = re.sub(r"\s+", " ", data["medical_history"]).strip()

        logger.debug(f"Extracted data: {data}")
        return data
    except Exception as e:
        logger.error(f"Error in extract_data_from_pdf: {str(e)}")
        raise

def simplify_text(text: str) -> str:
    try:
        try:
            data = json.loads(text)
            output = "Simplified Care Plan\n\n"
            if "Summary" in data:
                output += "Brief Summary:\n" + data["Summary"] + "\n\n"
            if "Schedule" in data:
                output += "Daily Schedule:\n" + "\n".join([f"- {s}" for s in data["Schedule"]]) + "\n\n"
            if "Medications" in data:
                output += "Medications:\n" + "\n".join([f"- {m}" for m in data["Medications"]]) + "\n\n"
            if "Tasks" in data:
                output += "Tasks:\n" + "\n".join([f"- {t}" for t in data["Tasks"]]) + "\n\n"
            return output.strip()
        except json.JSONDecodeError:
            replacements = {
                "antihypertensive": "blood pressure medicine",
                "daily activities": "things to do each day",
                "dietary recommendations": "food advice",
                "administer": "give",
                "hypertension": "high blood pressure",
                "medication regimen": "medicine schedule",
                "monitor": "check",
                "assess": "look at",
                "ambulation": "walking",
                "bid": "twice a day",
                "prn": "as needed",
                "metformin": "diabetes medicine",
                "amlodipine": "blood pressure medicine",
                "aspirin": "blood thinner",
                "paracetamol": "pain reliever"
            }
            simplified = text.lower()
            for old, new in replacements.items():
                simplified = simplified.replace(old.lower(), new)

            schedule = []
            tasks = []
            medications = []
            summary = "Patient requires support for chronic conditions and mobility."

            schedule_pattern = r"(?:give|take|check|do).+?(?:daily at|twice a day|as needed|at \d{1,2}(?::\d{2})?\s*(?:am|pm)?)"
            for match in re.finditer(schedule_pattern, simplified, re.IGNORECASE):
                schedule.append(match.group(0).strip())

            med_pattern = r"(?:blood pressure medicine|diabetes medicine|blood thinner|pain reliever|medicine|[a-z]+?\s*(?:medicine|tablet|pill)).*?(?:daily|twice a day|as needed)"
            for match in re.finditer(med_pattern, simplified, re.IGNORECASE):
                medications.append(match.group(0).strip())

            task_pattern = r"(?:help with|do|follow|things to do each day|walking|exercise|food advice|meal preparation|carer support).+?(?:\n|$|\.)"
            for match in re.finditer(task_pattern, simplified, re.IGNORECASE):
                tasks.append(match.group(0).strip())

            schedule = list(dict.fromkeys(schedule))
            medications = list(dict.fromkeys(medications))
            tasks = list(dict.fromkeys(tasks))

            output = "Simplified Care Plan\n\n"
            output += "Brief Summary:\n" + summary + "\n\n"
            if schedule:
                output += "Daily Schedule:\n" + "\n".join([f"- {s}" for s in schedule]) + "\n\n"
            if medications:
                output += "Medications:\n" + "\n".join([f"- {m}" for m in medications]) + "\n\n"
            if tasks:
                output += "Tasks:\n" + "\n".join([f"- {t}" for t in tasks]) + "\n\n"

            if not tasks:
                tasks = [
                    "Give medicines as listed in the schedule.",
                    "Follow food advice, like eating healthy meals.",
                    "Help with daily activities, like wheelchair mobility."
                ]
                output += "Tasks:\n" + "\n".join([f"- {t}" for t in tasks]) + "\n"

            return output.strip()
    except Exception as e:
        logger.error(f"Error in simplify_text: {str(e)}")
        raise

def translate_text(text: str, target_lang: str) -> str:
    try:
        url = "https://api-free.deepl.com/v2/translate"
        params = {
            "auth_key": DEEPL_API_KEY,
            "text": text,
            "target_lang": target_lang
        }
        response = requests.post(url, data=params)
        if response.status_code == 200:
            return response.json()["translations"][0]["text"]
        logger.error(f"DeepL API error: {response.status_code} - {response.text}")
        return "Translation failed"
    except Exception as e:
        logger.error(f"Error in translate_text: {str(e)}")
        raise

def create_pdf(patient_name: str, care_plan: str, simplified_plan: str, translated_plan: str = None) -> str:
    try:
        pdf_path = f"care_plan_{patient_name.replace(' ', '_')}.pdf"
        c = canvas.Canvas(pdf_path, pagesize=letter)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, 780, f"Care Plan for {patient_name}")
        
        y = 750
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Original Care Plan")
        y -= 20
        c.setFont("Helvetica", 10)
        text_obj = c.beginText(50, y)
        text_obj.setFont("Helvetica", 10)
        try:
            data = json.loads(care_plan)
            text_obj.textLine("Brief Summary:")
            text_obj.textLine(data.get("Summary", "Patient requires support for chronic conditions and mobility."))
            text_obj.textLine("")
            if "Schedule" in data:
                text_obj.textLine("Daily Schedule:")
                for item in data["Schedule"]:
                    text_obj.textLine(f"- {item}")
                text_obj.textLine("")
            if "Medications" in data:
                text_obj.textLine("Medications:")
                for item in data["Medications"]:
                    text_obj.textLine(f"- {item}")
                text_obj.textLine("")
            if "Tasks" in data:
                text_obj.textLine("Tasks:")
                for item in data["Tasks"]:
                    text_obj.textLine(f"- {item}")
        except json.JSONDecodeError:
            for line in care_plan.split('\n'):
                text_obj.textLine(line[:100])
        c.drawText(text_obj)
        y = text_obj.getY() - 20

        if y < 150:
            c.showPage()
            y = 750
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y, "Simplified Plan (English)")
        y -= 20
        text_obj = c.beginText(50, y)
        text_obj.setFont("Helvetica", 10)
        for line in simplified_plan.split('\n'):
            if y < 50:
                c.drawText(text_obj)
                c.showPage()
                y = 750
                text_obj = c.beginText(50, y)
                text_obj.setFont("Helvetica", 10)
            text_obj.textLine(line[:100])
            y -= 15
        c.drawText(text_obj)

        if translated_plan:
            if y < 150:
                c.showPage()
                y = 750
            else:
                y -= 20
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, f"Translated Plan (Polish)")
            y -= 20
            text_obj = c.beginText(50, y)
            text_obj.setFont("Helvetica", 10)
            for line in translated_plan.split('\n'):
                if y < 50:
                    c.drawText(text_obj)
                    c.showPage()
                    y = 750
                    text_obj = c.beginText(50, y)
                    text_obj.setFont("Helvetica", 10)
                text_obj.textLine(line[:100])
                y -= 15
            c.drawText(text_obj)

        c.save()
        return pdf_path
    except Exception as e:
        logger.error(f"Error in create_pdf: {str(e)}")
        raise

def extract_pdf_text(file: UploadFile) -> str:
    try:
        pdf_reader = PdfReader(io.BytesIO(file.file.read()))
        text = ""
        for page in pdf_reader.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + "\n"
        return text if text else "No text extracted"
    except Exception as e:
        logger.error(f"Error in extract_pdf_text: {str(e)}")
        raise

@app.get("/health")
def health_check():
    return {"status": "OK"}

@app.post("/generate_plan")
async def generate_plan(
    patient_name: str = Form(...),
    date_of_birth: str = Form(...),
    gender: str = Form(...),
    nhs_number: str = Form(...),
    date_of_assessment: str = Form(...),
    medical_history: str = Form(...),
    mobility: str = Form(...),
    language: str = Form(None)
):
    logger.info(f"Generating plan for {patient_name}, language: {language}")
    try:
        medical_history_truncated = medical_history[:300] + "..." if len(medical_history) > 300 else medical_history
        prompt = (
            f"You are a medical professional creating a care plan. Generate a structured care plan in JSON format for a patient named {patient_name}, "
            f"born {date_of_birth}, gender {gender}, NHS number {nhs_number}, assessed on {date_of_assessment}, "
            f"with medical history: {medical_history_truncated}, and mobility: {mobility}. "
            f"Include the following sections: 'Summary' (brief overview of patient's condition and care needs), "
            f"'Schedule' (medication and task times), 'Medications' (drug names, dosages, and instructions), "
            f"'Tasks' (daily activities and caregiver instructions). Ensure accuracy, clarity, and JSON format."
        )
        logger.debug("Generating text with Hugging Face...")
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 300, "temperature": 0.7, "top_p": 0.9}
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                break
            except (ReadTimeout, ConnectionError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Hugging Face API failed after {max_retries} attempts: {str(e)}")
                sleep(2 ** attempt)
            except HTTPError as e:
                logger.error(f"Hugging Face API HTTP error: {str(e)} - Response: {response.text}")
                raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")
        generated_text = response.json()[0]["generated_text"]
        logger.debug("Simplifying text...")
        simplified_plan = simplify_text(generated_text)
        translated_plan = None
        if language and language != "None":
            logger.debug(f"Translating to {language}...")
            translated_plan = translate_text(simplified_plan, language)
        logger.debug("Creating PDF...")
        pdf_path = create_pdf(patient_name, generated_text, simplified_plan, translated_plan)
        response = {
            "patient_name": patient_name,
            "date_of_birth": date_of_birth,
            "gender": gender,
            "nhs_number": nhs_number,
            "date_of_assessment": date_of_assessment,
            "medical_history": medical_history,
            "mobility": mobility,
            "care_plan": generated_text,
            "simplified_plan": simplified_plan,
            "pdf_url": f"/download_pdf/{pdf_path}"
        }
        if translated_plan:
            response["translated_plan"] = translated_plan
        logger.info("Plan generated successfully")
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error in generate_plan: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": f"Failed to generate care plan: {str(e)}"}, status_code=500)

@app.post("/upload_plan")
async def upload_plan(
    patient_name: str = Form(None),
    file: UploadFile = File(...),
    language: str = Form(None)
):
    logger.info(f"Uploading plan, patient_name: {patient_name}, language: {language}")
    try:
        if not file.filename.endswith('.pdf'):
            logger.error("Unsupported file format")
            return JSONResponse(content={"error": "Only PDF files are supported"}, status_code=400)
        
        logger.debug("Extracting PDF text...")
        pdf_text = extract_pdf_text(file)
        logger.debug("Extracting structured data...")
        extracted_data = extract_data_from_pdf(pdf_text)

        patient_name = patient_name or extracted_data["patient_name"]
        if patient_name == "Unknown":
            logger.error("Patient name not provided or extracted")
            return JSONResponse(content={"error": "Patient name is required"}, status_code=400)

        medical_history_truncated = extracted_data["medical_history"][:300] + "..." if len(extracted_data["medical_history"]) > 300 else extracted_data["medical_history"]
        prompt = (
            f"You are a medical professional creating a care plan. Generate a structured care plan in JSON format for a patient named {patient_name}, "
            f"born {extracted_data['date_of_birth']}, gender {extracted_data['gender']}, NHS number {extracted_data['nhs_number']}, "
            f"assessed on {extracted_data['date_of_assessment']}, with medical history: {medical_history_truncated}, "
            f"and mobility: {extracted_data['mobility']}. "
            f"Include the following sections: 'Summary' (brief overview of patient's condition and care needs), "
            f"'Schedule' (medication and task times), 'Medications' (drug names, dosages, and instructions), "
            f"'Tasks' (daily activities and caregiver instructions). Ensure accuracy, clarity, and JSON format."
        )
        logger.debug("Generating text with Hugging Face...")
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 300, "temperature": 0.7, "top_p": 0.9}
        }
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(
                    "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
                response.raise_for_status()
                break
            except (ReadTimeout, ConnectionError) as e:
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
                if attempt == max_retries - 1:
                    raise Exception(f"Hugging Face API failed after {max_retries} attempts: {str(e)}")
                sleep(2 ** attempt)
            except HTTPError as e:
                logger.error(f"Hugging Face API HTTP error: {str(e)} - Response: {response.text}")
                raise Exception(f"Hugging Face API error: {response.status_code} - {response.text}")
        generated_text = response.json()[0]["generated_text"]
        logger.debug("Simplifying text...")
        simplified_plan = simplify_text(generated_text)
        translated_plan = None
        if language and language != "None":
            logger.debug(f"Translating to {language}...")
            translated_plan = translate_text(simplified_plan, language)
        logger.debug("Creating PDF...")
        pdf_path = create_pdf(patient_name, generated_text, simplified_plan, translated_plan)
        response = {
            "patient_name": patient_name,
            "date_of_birth": extracted_data["date_of_birth"],
            "gender": extracted_data["gender"],
            "nhs_number": extracted_data["nhs_number"],
            "date_of_assessment": extracted_data["date_of_assessment"],
            "medical_history": extracted_data["medical_history"],
            "mobility": extracted_data["mobility"],
            "care_plan": generated_text,
            "simplified_plan": simplified_plan,
            "pdf_url": f"/download_pdf/{pdf_path}"
        }
        if translated_plan:
            response["translated_plan"] = translated_plan
        logger.info("Plan generated from PDF successfully")
        return JSONResponse(content=response)
    except Exception as e:
        logger.error(f"Error in upload_plan: {str(e)}", exc_info=True)
        return JSONResponse(content={"error": f"Failed to process uploaded plan: {str(e)}"}, status_code=500)

@app.get("/download_pdf/{pdf_path:path}")
async def download_pdf(pdf_path: str):
    try:
        if os.path.exists(pdf_path):
            return FileResponse(pdf_path, media_type="application/pdf", filename=pdf_path)
        logger.error(f"PDF not found: {pdf_path}")
        return JSONResponse(content={"error": "PDF not found"}, status_code=404)
    except Exception as e:
        logger.error(f"Error in download_pdf: {str(e)}")
        return JSONResponse(content={"error": f"Failed to download PDF: {str(e)}"}, status_code=500)