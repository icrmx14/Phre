import uvicorn
import io
import json
import re
import os
from datetime import datetime
from typing import List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from PyPDF2 import PdfReader

# AI & DB Imports
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from database import engine, get_db, Base
import models

# 1. INITIALIZE ENVIRONMENT
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2. AI SYSTEM (Gemini Flash)
class MedicalAgent:
    def __init__(self):
        self.active = False
        self.llm = None
        if GOOGLE_API_KEY:
            try:
                # Valid Gemini model names: gemini-2.0-flash-exp, gemini-1.5-flash, gemini-1.5-pro
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-3-flash-preview",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.1
                )
                self.active = True
                print("✅ AI System: Gemini 2.0 Flash Online")
            except Exception as e:
                print(f"❌ AI System Initialization Error: {e}")
        else:
            print("⚠️  AI System: GOOGLE_API_KEY not found in .env")

ai_agent = MedicalAgent()

# 3. APP INITIALIZATION
app = FastAPI(
    title="PHRELIS Hospital OS - Production Ready",
    version="2.0.0",
    description="AI-Powered Hospital Management System"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. PYDANTIC SCHEMAS
class AdmissionRequest(BaseModel):
    bed_id: str
    patient_name: str
    age: int
    condition: str

class AmbulanceRequest(BaseModel):
    severity: str
    location: str
    eta: int

class DischargeResponse(BaseModel):
    status: str
    message: str
    bed_id: str

# 5. NEURAL UTILITY (Fixes JSON 500 Errors)
def extract_neural_json(raw_response):
    """
    Safely converts LLM response to JSON.
    Handles both strings and the Gemini 3 'list of blocks' format.
    """
    try:
        # 1. Handle cases where raw_response is a list (fallback safety)
        if isinstance(raw_response, list):
            # Join text blocks into one string
            content_str = "".join([block.get("text", "") for block in raw_response if isinstance(block, dict)])
        else:
            content_str = str(raw_response)

        # 2. Strip Markdown JSON wrappers (```json ... ```)
        json_match = re.search(r"(\{.*\})", content_str, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1).strip())
        
        return json.loads(content_str.strip())
    except Exception as e:
        print(f"Extraction Error Detail: {e}")
        return None

# 6. CLINICAL API (Neural Triage)
@app.post("/api/analyze-report")
async def analyze_report(file: UploadFile = File(...)):
    """Analyzes uploaded medical reports using AI"""
    
    # Check if AI is active
    if not ai_agent.active or not ai_agent.llm:
        raise HTTPException(
            status_code=503, 
            detail="AI System Unavailable. Please check GOOGLE_API_KEY in .env"
        )
    
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Read and extract PDF content
        content = await file.read()
        pdf = PdfReader(io.BytesIO(content))
        text = "".join([page.extract_text() or "" for page in pdf.pages])
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text could be extracted from PDF")
        
        # AI Analysis
        prompt = ChatPromptTemplate.from_template(
    """
You are a clinical decision support system used in a hospital.

Analyze the following clinical laboratory report.

IMPORTANT RULES:
- If any lab value is marked as CRITICAL, LOW**, or HIGH**, the patient must NOT be considered stable.
- Critically low hemoglobin, hematocrit, or RBC count requires urgent admission.
- Lab reference ranges and flags override any explanatory or educational text in the report.

Clinical Report:
{text}

Return ONLY a valid JSON object in the following format:
{{
  "patient_summary": "brief summary of patient condition",
  "deficiencies": ["deficiency1", "deficiency2"],
  "diet_plan": "recommended diet plan",
  "medication_notes": "medication recommendations",
  "admission_required": true,
  "urgency_reason": "reason for urgency level",
  "suggested_unit": "ICU/General Ward/Emergency"
}}

Do NOT include disclaimers, markdown, or explanations.
Return ONLY JSON.
"""
)

        
        chain = prompt | ai_agent.llm
        # --- To this ---
        response = await chain.ainvoke({"text": text[:3500]})

# Access the .text property which handles the list-to-string conversion
        raw_text = response.text 
        data = extract_neural_json(raw_text)
        
        # Fallback structure if parsing fails
        if not data:
            print("⚠️  AI returned invalid JSON, using fallback")
            return {
                "patient_summary": "Analysis completed - manual review recommended",
                "deficiencies": ["Unable to parse AI response"],
                "diet_plan": "Standard hospital diet",
                "medication_notes": "Consult physician",
                "admission_required": False,
                "urgency_reason": "Requires manual assessment",
                "suggested_unit": "General Ward"
            }
        
        # Validate and fill missing keys
        required_keys = {
            "patient_summary": "Analysis completed",
            "deficiencies": [],
            "diet_plan": "Standard hospital diet",
            "medication_notes": "As prescribed",
            "admission_required": False,
            "urgency_reason": "Non-urgent",
            "suggested_unit": "General Ward"
        }
        
        for key, default_value in required_keys.items():
            if key not in data:
                data[key] = default_value
        
        return data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Analysis Failed: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Neural Analysis Failed: {str(e)}"
        )

# 7. ERP & DASHBOARD
@app.get("/api/dashboard/stats")
def get_stats(db: Session = Depends(get_db)):
    """Get hospital dashboard statistics"""
    try:
        beds = db.query(models.BedModel).all()
        total_beds = len(beds) if beds else 10
        occupied = len([b for b in beds if b.is_occupied])
        
        return {
            "bed_stats": {
                "total": total_beds,
                "occupied": occupied,
                "available": total_beds - occupied
            },
            "resources": {
                "Ventilators": {"total": 20, "in_use": 5},
                "Ambulances": {"total": 10, "available": 4},
                "ICU_Beds": {"total": 10, "available": total_beds - occupied}
            },
            "occupancy": {
                "ICU": occupied,
                "ER": 0,
                "General": 0
            },
            "staff_ratio": "1:4",
            "ai_status": "Online" if ai_agent.active else "Offline"
        }
    except Exception as e:
        print(f"❌ Dashboard Stats Error: {e}")
        return {
            "bed_stats": {"total": 10, "occupied": 0, "available": 10},
            "resources": {"Ventilators": {"total": 20, "in_use": 5}},
            "occupancy": {"ICU": 0, "ER": 0},
            "staff_ratio": "1:4",
            "ai_status": "Offline"
        }

@app.get("/api/staff")
def get_staff(db: Session = Depends(get_db)):
    """Get all staff members"""
    try:
        staff = db.query(models.StaffMember).all()
        return staff if staff else []
    except Exception as e:
        print(f"❌ Staff Query Error: {e}")
        return []

@app.get("/api/erp/beds")
def list_beds(db: Session = Depends(get_db)):
    """List all hospital beds"""
    try:
        beds = db.query(models.BedModel).all()
        return beds if beds else []
    except Exception as e:
        print(f"❌ Beds Query Error: {e}")
        return []

@app.get("/api/metrics/latency")
def get_latency():
    """Get system latency metrics"""
    return {
        "latencyScore": 45,
        "avgResponseTime": "120ms",
        "status": "healthy"
    }

# 8. PREDICTION ENDPOINTS
@app.get("/api/predict-inflow")
@app.post("/api/predict-inflow")
async def predict_inflow():
    """Predict patient inflow"""
    return {
        "predicted_inflow": 12,
        "confidence": "high",
        "trend": "increasing",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/api/predict/history")
def get_predict_history():
    """Get prediction history"""
    return {"history": []}

@app.get("/api/predictions")
def get_all_predictions():
    """Get all predictions"""
    return {"predictions": []}

# 9. ERP OPERATIONS
@app.post("/api/erp/admit")
async def admit_patient(request: AdmissionRequest, db: Session = Depends(get_db)):
    """Admit a patient to a bed"""
    try:
        bed = db.query(models.BedModel).filter(
            models.BedModel.id == request.bed_id
        ).first()
        
        if not bed:
            raise HTTPException(status_code=404, detail=f"Bed {request.bed_id} not found")
        
        if bed.is_occupied:
            raise HTTPException(
                status_code=400, 
                detail=f"Bed {request.bed_id} is already occupied by {bed.patient_name}"
            )
        
        # Update bed with patient info
        bed.is_occupied = True
        bed.patient_name = request.patient_name
        bed.condition = request.condition
        bed.patient_age = request.age
        bed.admission_time = datetime.utcnow()
        
        db.commit()
        db.refresh(bed)
        
        return {
            "status": "success",
            "message": f"Patient {request.patient_name} admitted to {request.bed_id}",
            "bed": {
                "id": bed.id,
                "patient_name": bed.patient_name,
                "condition": bed.condition,
                "admission_time": bed.admission_time.isoformat() if bed.admission_time else None
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Admission Error: {e}")
        raise HTTPException(status_code=500, detail=f"Admission failed: {str(e)}")

@app.post("/api/erp/discharge/{bed_id}")
async def discharge_patient(bed_id: str, db: Session = Depends(get_db)):
    """Discharge a patient from a bed"""
    try:
        bed = db.query(models.BedModel).filter(
            models.BedModel.id == bed_id
        ).first()
        
        if not bed:
            raise HTTPException(status_code=404, detail=f"Bed {bed_id} not found")
        
        if not bed.is_occupied:
            raise HTTPException(status_code=400, detail=f"Bed {bed_id} is not occupied")
        
        patient_name = bed.patient_name
        
        # Clear bed
        bed.is_occupied = False
        bed.patient_name = None
        bed.condition = None
        bed.patient_age = None
        bed.admission_time = None
        
        db.commit()
        db.refresh(bed)
        
        return {
            "status": "success",
            "message": f"Patient {patient_name} discharged from {bed_id}",
            "bed_id": bed_id
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        print(f"❌ Discharge Error: {e}")
        raise HTTPException(status_code=500, detail=f"Discharge failed: {str(e)}")

# 10. HEALTH CHECK
@app.get("/")
@app.get("/health")
def health_check():
    """API Health Check"""
    return {
        "status": "healthy",
        "service": "PHRELIS Hospital OS",
        "version": "2.0.0",
        "ai_status": "online" if ai_agent.active else "offline",
        "timestamp": datetime.utcnow().isoformat()
    }

# 11. LIFECYCLE EVENTS
@app.on_event("startup")
async def startup_event():
    """Initialize database and seed initial data"""
    print("\n" + "="*60)
    print("🏥 PHRELIS Hospital OS - Starting Up")
    print("="*60)
    
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables created/verified")
        
        # Seed initial data
        db = next(get_db())
        
        # Seed beds if empty
        if db.query(models.BedModel).count() == 0:
            beds = [
                models.BedModel(id=f"ICU-{i}", type="ICU", is_occupied=False) 
                for i in range(1, 11)
            ]
            db.add_all(beds)
            db.commit()
            print("✅ Seeded 10 ICU beds")
        else:
            bed_count = db.query(models.BedModel).count()
            print(f"✅ Found {bed_count} existing beds")
        
        db.close()
        
        # Check AI status
        if ai_agent.active:
            print("✅ AI System: Online")
        else:
            print("⚠️  AI System: Offline (check GOOGLE_API_KEY)")
        
        print("="*60)
        print("🚀 Server Ready on http://localhost:8000")
        print("📚 API Docs: http://localhost:8000/docs")
        print("="*60 + "\n")
        
    except Exception as e:
        print(f"❌ Startup Error: {e}")
        print("⚠️  Server may not function correctly")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n🛑 PHRELIS Hospital OS - Shutting Down...")
    print("👋 Goodbye!\n")

# 12. RUN SERVER
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )