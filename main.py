# === AGEI Call Analysis Python Webhook ===
from flask import Flask, request, jsonify
import requests, os, openai
import tempfile, subprocess
import logging
import re
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# === CONFIG ===
openai.api_key = os.environ['OPENAI_API_KEY']
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']
SHEETS_SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE_ACCOUNT_FILE = 'service_account.json'

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Column mappings matching Google Sheets
COLUMNS = {
    'PATIENT_NAME': 1,        # A
    'MRN': 2,                 # B
    'CALL_SCENARIO': 3,       # C
    'APPOINTMENT_TYPE': 4,    # D
    'APPOINTMENT_REASON': 5,  # E
    'FILENAME': 6,            # F
    'INITIAL_QUESTIONS': 7,   # G
    'INITIAL_RESPONSES': 8,   # H
    'DEMOGRAPHICS': 9,        # I
    'SCHEDULING_QUESTIONS': 10, # J
    'SCHEDULING_RESPONSES': 11, # K
    'FINAL_STEPS': 12,        # L
    'SUMMARY': 13,            # M
    'TRANSCRIPT_LINK': 14,    # N
    'STATUS': 15              # O
}

# === HELPER FUNCTIONS ===
def download_file(file_url, token):
    """Download file with better error handling and timeout"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(file_url, headers=headers, timeout=300)  # 5 min timeout
        r.raise_for_status()
        
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tmp_file.write(r.content)
        tmp_file.close()
        
        logger.info(f"Downloaded file: {tmp_file.name}, size: {len(r.content)} bytes")
        return tmp_file.name
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download file: {e}")
        raise Exception(f"File download failed: {str(e)}")

def split_channels(mp3_path):
    """Split stereo audio with better error handling"""
    try:
        base = mp3_path.replace(".mp3", "")
        left_path = base + "_left.wav"
        right_path = base + "_right.wav"
        
        cmd = [
            "ffmpeg", "-y", "-i", mp3_path,
            "-filter_complex", "[0:a]channelsplit=channel_layout=stereo[left][right]",
            "-map", "[left]", left_path,
            "-map", "[right]", right_path
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"Successfully split channels: {left_path}, {right_path}")
        return left_path, right_path
        
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg failed: {e.stderr}")
        raise Exception(f"Audio processing failed: {e.stderr}")

def transcribe(file_path):
    """Transcribe with better error handling"""
    try:
        with open(file_path, "rb") as f:
            response = openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
        logger.info(f"Transcribed file: {file_path}")
        return response
    except Exception as e:
        logger.error(f"Transcription failed for {file_path}: {e}")
        raise Exception(f"Transcription failed: {str(e)}")

def reconstruct_conversation(agent_text, patient_text):
    """Reconstruct conversation with better prompt"""
    try:
        prompt = f"""
You are reconstructing a phone conversation between a medical practice scheduling agent and a patient.

Agent transcript (may be out of order):
{agent_text}

Patient transcript (may be out of order):
{patient_text}

Instructions:
1. Reconstruct the conversation in chronological order
2. Use exactly this format for each line:
   Agent: [what the agent said]
   Patient: [what the patient said]
3. If you can't determine the exact order, use context clues
4. Do not add any commentary, just the reconstructed dialogue

Reconstructed conversation:
"""
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a conversation reconstruction specialist. Only output the reconstructed dialogue using the Agent:/Patient: format."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        logger.info("Successfully reconstructed conversation")
        return result
        
    except Exception as e:
        logger.error(f"Conversation reconstruction failed: {e}")
        raise Exception(f"Reconstruction failed: {str(e)}")

def analyze_scheduling_workflow(transcript, patient_context):
    """Analyze the scheduling workflow from the call transcript"""
    try:
        analysis_prompt = f"""
You are analyzing a medical practice scheduling call to understand the workflow and qualifying questions used.

PATIENT CONTEXT:
- Patient Name: {patient_context.get('patient_name', 'Unknown')}
- MRN: {patient_context.get('mrn', 'Unknown')}
- Call Type: {patient_context.get('call_scenario', 'Unknown')}
- Final Appointment Type: {patient_context.get('appointment_type', 'Unknown')}
- Final Appointment Reason: {patient_context.get('appointment_reason', 'Unknown')}

CALL TRANSCRIPT:
{transcript}

ANALYSIS INSTRUCTIONS:
Please analyze this call and extract information for each workflow phase. For each section, focus on WHAT was asked/discussed, not how well it was done.

Return your analysis in this EXACT format:

**INITIAL_QUESTIONS:**
[List the qualifying questions the agent asked to understand the patient's needs - e.g., "What symptoms are you experiencing?", "Who referred you?", "What type of consultation do you need?"]

**INITIAL_RESPONSES:**
[List the patient's key responses to those questions - e.g., "Dry eyes and blurry vision", "Self-referred", "Eye exam for cataract concerns"]

**DEMOGRAPHICS:**
[List what demographic/contact information was collected - e.g., "Name, date of birth, phone number, address, insurance information"]

**SCHEDULING_QUESTIONS:**
[List questions asked about scheduling preferences - e.g., "What location do you prefer?", "What days work best for you?", "Morning or afternoon preference?"]

**SCHEDULING_RESPONSES:**
[List the patient's scheduling preferences - e.g., "Main office location", "Weekdays only", "Morning appointments preferred"]

**FINAL_STEPS:**
[List administrative/final steps discussed - e.g., "Provided parking information", "Mentioned intake paperwork", "Confirmed appointment details", "Gave office phone number"]

**SUMMARY:**
[2-3 sentence summary of the overall call and outcome]

Focus on documenting the actual workflow used, not evaluating effectiveness.
"""
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical scheduling workflow analyst. Extract the specific questions asked and responses given during each phase of the scheduling process."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        logger.info("Successfully analyzed scheduling workflow")
        return result
        
    except Exception as e:
        logger.error(f"Workflow analysis failed: {e}")
        raise Exception(f"Analysis failed: {str(e)}")

def parse_workflow_analysis(analysis_text):
    """Parse the structured workflow analysis into separate fields"""
    fields = {
        "INITIAL_QUESTIONS": "",
        "INITIAL_RESPONSES": "",
        "DEMOGRAPHICS": "",
        "SCHEDULING_QUESTIONS": "",
        "SCHEDULING_RESPONSES": "",
        "FINAL_STEPS": "",
        "SUMMARY": ""
    }
    
    # Use regex to extract fields
    for field_name in fields.keys():
        pattern = rf'\*\*{re.escape(field_name)}:\*\*\s*(.+?)(?=\*\*|\Z)'
        match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
        if match:
            # Clean up the extracted text
            extracted = match.group(1).strip()
            # Remove leading brackets or formatting
            extracted = re.sub(r'^\[|\]$', '', extracted)
            fields[field_name] = extracted
        else:
            logger.warning(f"Could not extract field: {field_name}")
    
    return fields

def upload_to_drive(file_path, folder_id):
    """Upload to Drive with better error handling"""
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=DRIVE_SCOPES
        )
        service = build('drive', 'v3', credentials=creds)
        
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        
        media = MediaFileUpload(file_path, mimetype='text/plain')
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id,webViewLink',
            supportsAllDrives=True
        ).execute()
        
        logger.info(f"Uploaded file to Drive: {file['webViewLink']}")
        return file['webViewLink']
        
    except Exception as e:
        logger.error(f"Drive upload failed: {e}")
        raise Exception(f"Upload failed: {str(e)}")

def update_sheet(sheet_id, sheet_name, row_number, transcript_url, workflow_fields):
    """Update the Google Sheet with analysis results"""
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SHEETS_SCOPES
        )
        service = build('sheets', 'v4', credentials=creds)
        
        # Prepare the values for columns G through O
        values = [
            workflow_fields["INITIAL_QUESTIONS"],     # G
            workflow_fields["INITIAL_RESPONSES"],     # H
            workflow_fields["DEMOGRAPHICS"],          # I
            workflow_fields["SCHEDULING_QUESTIONS"],  # J
            workflow_fields["SCHEDULING_RESPONSES"],  # K
            workflow_fields["FINAL_STEPS"],          # L
            workflow_fields["SUMMARY"],              # M
            transcript_url,                          # N
            "Completed"                              # O
        ]
        
        # Update range from G to O for the specific row
        update_range = f"{sheet_name}!G{row_number}:O{row_number}"
        
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=update_range,
            valueInputOption="RAW",
            body={"values": [values]}
        ).execute()
        
        logger.info(f"Updated sheet row {row_number} successfully")
        return True
        
    except Exception as e:
        logger.error(f"Sheet update failed: {e}")
        raise Exception(f"Sheet update failed: {str(e)}")

def analyze_patterns(calls_data):
    """Analyze patterns across multiple calls"""
    try:
        # Prepare data for analysis
        calls_summary = []
        for call in calls_data:
            summary = f"""
Call {call['row_number']}:
- Appointment Type: {call['appointment_type']}
- Appointment Reason: {call['appointment_reason']}
- Call Scenario: {call['call_scenario']}
- Initial Questions: {call['initial_questions']}
- Initial Responses: {call['initial_responses']}
- Demographics Collected: {call['demographics']}
- Scheduling Questions: {call['scheduling_questions']}
- Scheduling Responses: {call['scheduling_responses']}
- Final Steps: {call['final_steps']}
"""
            calls_summary.append(summary)
        
        pattern_prompt = f"""
You are analyzing patterns across multiple medical practice scheduling calls to identify common workflows and decision-making processes.

CALLS DATA:
{chr(10).join(calls_summary)}

ANALYSIS INSTRUCTIONS:
Analyze these calls to identify patterns in:
1. What qualifying questions are commonly asked for different appointment types
2. How patient responses influence appointment type/reason selection
3. Common scheduling workflow patterns
4. Demographic information typically collected
5. Standard final steps/administrative tasks

Provide a comprehensive analysis report with:

**APPOINTMENT TYPE PATTERNS:**
[For each appointment type, what qualifying questions are typically asked and what patient responses lead to that appointment type]

**QUALIFYING QUESTIONS BY CONDITION:**
[Common qualifying questions grouped by patient condition/symptoms mentioned]

**SCHEDULING WORKFLOW PATTERNS:**
[Common sequences of questions and steps in the scheduling process]

**DEMOGRAPHIC COLLECTION PATTERNS:**
[What demographic information is consistently collected across calls]

**FINAL STEPS PATTERNS:**
[Standard administrative steps and information provided to patients]

**KEY INSIGHTS:**
[Notable patterns, trends, or decision-making processes observed]

**RECOMMENDATIONS:**
[Suggestions for standardizing or improving the scheduling workflow based on patterns observed]

Focus on actionable insights that could help improve or standardize the scheduling process.
"""
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a healthcare operations analyst specializing in scheduling workflows. Identify actionable patterns and insights."},
                {"role": "user", "content": pattern_prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        logger.info(f"Successfully analyzed patterns for {len(calls_data)} calls")
        return result
        
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        raise Exception(f"Pattern analysis failed: {str(e)}")

def save_pattern_analysis(sheet_id, pattern_analysis, calls_analyzed):
    """Save pattern analysis to a separate sheet"""
    try:
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_FILE, scopes=SHEETS_SCOPES
        )
        service = build('sheets', 'v4', credentials=creds)
        
        # Try to create or update the Patterns Analysis sheet
        try:
            # Check if sheet exists
            sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
            sheets = sheet_metadata.get('sheets', [])
            pattern_sheet_exists = any(sheet['properties']['title'] == 'Patterns Analysis' for sheet in sheets)
            
            if not pattern_sheet_exists:
                # Create the sheet
                request = {
                    'addSheet': {
                        'properties': {
                            'title': 'Patterns Analysis'
                        }
                    }
                }
                service.spreadsheets().batchUpdate(
                    spreadsheetId=sheet_id,
                    body={'requests': [request]}
                ).execute()
                logger.info("Created 'Patterns Analysis' sheet")
        
        except Exception as e:
            logger.warning(f"Could not create Patterns Analysis sheet: {e}")
        
        # Prepare the data to insert
        timestamp = str(pd.Timestamp.now()) if 'pd' in globals() else str(datetime.now())
        header_row = [f"Pattern Analysis - {timestamp}", f"Calls Analyzed: {', '.join(map(str, calls_analyzed))}"]
        analysis_rows = pattern_analysis.split('\n')
        
        # Insert the data
        values = [header_row, []] + [[row] for row in analysis_rows]
        
        service.spreadsheets().values().append(
            spreadsheetId=sheet_id,
            range="Patterns Analysis!A:A",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": values}
        ).execute()
        
        logger.info("Successfully saved pattern analysis")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save pattern analysis: {e}")
        return False

def cleanup_files(*file_paths):
    """Safely cleanup temporary files"""
    for file_path in file_paths:
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up file: {file_path}")
            except Exception as e:
                logger.warning(f"Could not remove file {file_path}: {e}")

# === ENDPOINTS ===
@app.route('/')
def index():
    return "AGEI Call Analysis Webhook Service - Ready"

@app.route('/analyze', methods=['POST'])
def analyze_call():
    # Initialize file paths for cleanup
    mp3_path = None
    left_path = None
    right_path = None
    txt_path = None
    
    try:
        data = request.json
        
        # Handle test requests
        if data and data.get('test') == True:
            logger.info("Test request received")
            return jsonify({"status": "success", "message": "Webhook is working"})
        
        logger.info(f"Processing call: {data.get('file_name', 'unknown') if data else 'No data'}")
        
        # Validate required fields
        if not data:
            raise Exception("No JSON data received")
            
        required_fields = ['file_url', 'file_token', 'sheet_id', 'sheet_name', 
                          'row_number', 'transcript_folder_id', 'file_name']
        for field in required_fields:
            if field not in data:
                raise Exception(f"Missing required field: {field}")
        
        # Extract patient context
        patient_context = {
            'patient_name': data.get('patient_name', ''),
            'mrn': data.get('mrn', ''),
            'call_scenario': data.get('call_scenario', ''),
            'appointment_type': data.get('appointment_type', ''),
            'appointment_reason': data.get('appointment_reason', '')
        }
        
        # Download and process audio
        mp3_path = download_file(data['file_url'], data['file_token'])
        left_path, right_path = split_channels(mp3_path)
        
        # Transcribe both channels
        agent_text = transcribe(left_path)
        patient_text = transcribe(right_path)
        
        # Reconstruct conversation
        full_transcript = reconstruct_conversation(agent_text, patient_text)
        
        # Analyze scheduling workflow
        workflow_analysis = analyze_scheduling_workflow(full_transcript, patient_context)
        workflow_fields = parse_workflow_analysis(workflow_analysis)
        
        # Save transcript file
        txt_path = mp3_path.replace(".mp3", "_transcript.txt")
        with open(txt_path, "w", encoding='utf-8') as f:
            f.write("=== FULL CONVERSATION TRANSCRIPT ===\n\n")
            f.write(full_transcript)
            f.write("\n\n=== WORKFLOW ANALYSIS ===\n\n")
            f.write(workflow_analysis)
        
        # Upload to Drive and update sheet
        transcript_link = upload_to_drive(txt_path, data["transcript_folder_id"])
        update_sheet(data["sheet_id"], data["sheet_name"], data["row_number"], 
                    transcript_link, workflow_fields)
        
        logger.info(f"Successfully processed call: {data['file_name']}")
        return jsonify({
            "status": "success", 
            "transcript_link": transcript_link,
            "workflow_fields": workflow_fields
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing call: {error_msg}")
        
        # Try to update sheet with error status
        try:
            if 'data' in locals() and data:
                creds = service_account.Credentials.from_service_account_file(
                    SERVICE_ACCOUNT_FILE, scopes=SHEETS_SCOPES
                )
                service = build('sheets', 'v4', credentials=creds)
                
                error_range = f"{data['sheet_name']}!O{data['row_number']}"
                service.spreadsheets().values().update(
                    spreadsheetId=data["sheet_id"],
                    range=error_range,
                    valueInputOption="RAW",
                    body={"values": [[f"ERROR: {error_msg}"]]}
                ).execute()
        except:
            logger.error("Could not update sheet with error status")
        
        return jsonify({"error": error_msg}), 500
        
    finally:
        # Clean up all temporary files
        cleanup_files(mp3_path, left_path, right_path, txt_path)

@app.route('/analyze-patterns', methods=['POST'])
def analyze_patterns_endpoint():
    """Endpoint for pattern analysis across multiple calls"""
    try:
        data = request.json
        
        if not data or 'calls_data' not in data:
            raise Exception("No calls data provided")
        
        calls_data = data['calls_data']
        sheet_id = data.get('sheet_id')
        
        logger.info(f"Running pattern analysis on {len(calls_data)} calls")
        
        # Analyze patterns
        pattern_analysis = analyze_patterns(calls_data)
        
        # Save to sheet
        calls_analyzed = [call['row_number'] for call in calls_data]
        save_pattern_analysis(sheet_id, pattern_analysis, calls_analyzed)
        
        logger.info("Pattern analysis completed successfully")
        return jsonify({
            "status": "success",
            "calls_analyzed": len(calls_data),
            "pattern_analysis": pattern_analysis
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in pattern analysis: {error_msg}")
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)