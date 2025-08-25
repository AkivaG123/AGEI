# === AGEI Call Analysis Python Webhook - Enhanced with Decision Tree Analysis ===
from flask import Flask, request, jsonify
import requests, os, openai
import tempfile, subprocess
import logging
import re
import json
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# === CONFIG ===
openai.api_key = os.environ['OPENAI_API_KEY']
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive']
SHEETS_SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

# Build service account info from environment variables
def get_service_account_credentials(scopes):
    try:
        # Debug: Log which environment variables are available
        logger.info(f"Available environment variables: {list(os.environ.keys())}")
        
        # Check each required variable
        required_vars = ['GOOGLE_PROJECT_ID', 'GOOGLE_PRIVATE_KEY_ID', 'GOOGLE_PRIVATE_KEY', 'GOOGLE_CLIENT_EMAIL', 'GOOGLE_CLIENT_ID']
        for var in required_vars:
            if var not in os.environ:
                raise Exception(f"Missing environment variable: {var}")
            logger.info(f"{var} is present")
        
        service_account_info = {
            "type": "service_account",
            "project_id": os.environ['GOOGLE_PROJECT_ID'],
            "private_key_id": os.environ['GOOGLE_PRIVATE_KEY_ID'],
            "private_key": os.environ['GOOGLE_PRIVATE_KEY'].replace('\\n', '\n'),
            "client_email": os.environ['GOOGLE_CLIENT_EMAIL'],
            "client_id": os.environ['GOOGLE_CLIENT_ID'],
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": f"https://www.googleapis.com/robot/v1/metadata/x509/{os.environ['GOOGLE_CLIENT_EMAIL']}",
            "universe_domain": "googleapis.com"
        }
        
        logger.info("Successfully built service account info")
        return service_account.Credentials.from_service_account_info(service_account_info, scopes=scopes)
        
    except Exception as e:
        logger.error(f"Error creating credentials: {str(e)}")
        raise Exception(f"Credential creation failed: {str(e)}")

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
    'STATUS': 15,             # O
    'DECISION_SEQUENCE': 16,  # P
    'PATIENT_TYPE_DETERMINATION': 17, # Q
    'SYMPTOM_ASSESSMENT': 18, # R
    'PROVIDER_SELECTION_LOGIC': 19, # S
    'APPOINTMENT_TYPE_LOGIC': 20, # T
    'ROUTING_RULES': 21,      # U
    'DECISION_BRANCHES': 22   # V
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

def analyze_scheduling_workflow_enhanced(transcript, patient_context):
    """Enhanced analysis that includes both workflow and decision logic"""
    try:
        analysis_prompt = f"""
You are analyzing a medical practice scheduling call to understand both the workflow AND the decision-making logic used.

PATIENT CONTEXT:
- Patient Name: {patient_context.get('patient_name', 'Unknown')}
- MRN: {patient_context.get('mrn', 'Unknown')}
- Call Type: {patient_context.get('call_scenario', 'Unknown')}
- Final Appointment Type: {patient_context.get('appointment_type', 'Unknown')}
- Final Appointment Reason: {patient_context.get('appointment_reason', 'Unknown')}

CALL TRANSCRIPT:
{transcript}

ANALYSIS INSTRUCTIONS:
Analyze this call for both workflow documentation AND decision tree logic extraction.

Return your analysis in this EXACT format:

**INITIAL_QUESTIONS:**
[List the qualifying questions the agent asked to understand the patient's needs]

**INITIAL_RESPONSES:**
[List the patient's key responses to those questions]

**DEMOGRAPHICS:**
[List what demographic/contact information was collected]

**SCHEDULING_QUESTIONS:**
[List questions asked about scheduling preferences]

**SCHEDULING_RESPONSES:**
[List the patient's scheduling preferences]

**FINAL_STEPS:**
[List administrative/final steps discussed]

**SUMMARY:**
[2-3 sentence summary of the overall call and outcome]

**DECISION_SEQUENCE:**
[Map the logical flow: Question → Response → Next Action/Decision. Show how each answer led to the next step]

**PATIENT_TYPE_DETERMINATION:**
[How was patient type (new/existing) determined? What questions/responses led to this?]

**SYMPTOM_ASSESSMENT:**
[What symptoms were identified and how did they influence appointment routing?]

**PROVIDER_SELECTION_LOGIC:**
[What factors determined provider selection? Specialization, availability, symptoms, patient preference?]

**APPOINTMENT_TYPE_LOGIC:**
[What logic connected the patient's needs to the specific appointment type selected?]

**ROUTING_RULES:**
[What rules or protocols governed how this patient was handled? Emergency procedures, specialist requirements, etc.]

**DECISION_BRANCHES:**
[Identify key points where different patient responses would have led to different outcomes]

Focus on extracting both the workflow used AND the decision logic that could be replicated.
"""
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are both a workflow analyst and decision logic specialist. Extract both the process used and the logical decision-making patterns."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        logger.info("Successfully analyzed enhanced workflow and decision logic")
        return result
        
    except Exception as e:
        logger.error(f"Enhanced workflow analysis failed: {e}")
        raise Exception(f"Enhanced analysis failed: {str(e)}")

def parse_enhanced_analysis(analysis_text):
    """Parse the enhanced analysis into workflow and decision fields"""
    workflow_fields = {
        "INITIAL_QUESTIONS": "",
        "INITIAL_RESPONSES": "",
        "DEMOGRAPHICS": "",
        "SCHEDULING_QUESTIONS": "",
        "SCHEDULING_RESPONSES": "",
        "FINAL_STEPS": "",
        "SUMMARY": ""
    }
    
    decision_fields = {
        "DECISION_SEQUENCE": "",
        "PATIENT_TYPE_DETERMINATION": "",
        "SYMPTOM_ASSESSMENT": "",
        "PROVIDER_SELECTION_LOGIC": "",
        "APPOINTMENT_TYPE_LOGIC": "",
        "ROUTING_RULES": "",
        "DECISION_BRANCHES": ""
    }
    
    all_fields = {**workflow_fields, **decision_fields}
    
    # Use regex to extract fields
    for field_name in all_fields.keys():
        pattern = rf'\*\*{re.escape(field_name)}:\*\*\s*(.+?)(?=\*\*|\Z)'
        match = re.search(pattern, analysis_text, re.DOTALL | re.IGNORECASE)
        if match:
            extracted = match.group(1).strip()
            extracted = re.sub(r'^\[|\]$', '', extracted)
            all_fields[field_name] = extracted
        else:
            logger.warning(f"Could not extract field: {field_name}")
    
    return workflow_fields, decision_fields

def upload_to_drive(file_path, folder_id):
    """Upload to Drive with better error handling"""
    try:
        creds = get_service_account_credentials(DRIVE_SCOPES)
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

def update_sheet_enhanced(sheet_id, sheet_name, row_number, transcript_url, workflow_fields, decision_fields):
    """Update the Google Sheet with enhanced analysis results"""
    try:
        creds = get_service_account_credentials(SHEETS_SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        
        # First, ensure we have the decision logic columns (P through V)
        # Check current sheet structure and add columns if needed
        sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
        sheets = sheet_metadata.get('sheets', [])
        
        target_sheet = None
        for sheet in sheets:
            if sheet['properties']['title'] == sheet_name:
                target_sheet = sheet
                break
        
        if target_sheet:
            current_columns = target_sheet['properties']['gridProperties'].get('columnCount', 15)
            if current_columns < 22:  # We need columns A through V (22 columns)
                # Add more columns
                requests = [{
                    'appendDimension': {
                        'sheetId': target_sheet['properties']['sheetId'],
                        'dimension': 'COLUMNS',
                        'length': 22 - current_columns
                    }
                }]
                
                service.spreadsheets().batchUpdate(
                    spreadsheetId=sheet_id,
                    body={'requests': requests}
                ).execute()
                
                logger.info(f"Added columns to support decision logic analysis")
        
        # Prepare workflow values (columns G through O)
        workflow_values = [
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
        
        # Prepare decision logic values (columns P through V)
        decision_values = [
            decision_fields["DECISION_SEQUENCE"],        # P
            decision_fields["PATIENT_TYPE_DETERMINATION"], # Q
            decision_fields["SYMPTOM_ASSESSMENT"],       # R
            decision_fields["PROVIDER_SELECTION_LOGIC"], # S
            decision_fields["APPOINTMENT_TYPE_LOGIC"],   # T
            decision_fields["ROUTING_RULES"],            # U
            decision_fields["DECISION_BRANCHES"]         # V
        ]
        
        # Update workflow columns (G through O)
        workflow_range = f"{sheet_name}!G{row_number}:O{row_number}"
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=workflow_range,
            valueInputOption="RAW",
            body={"values": [workflow_values]}
        ).execute()
        
        logger.info(f"Workflow update completed successfully")        
        
        # Update decision logic columns (P through V)
        decision_range = f"{sheet_name}!P{row_number}:V{row_number}"
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=decision_range,
            valueInputOption="RAW",
            body={"values": [decision_values]}
        ).execute()

        logger.info(f"Decision logic update completed successfully")
        
        logger.info(f"Updated sheet row {row_number} with enhanced analysis")
        return True
        
    except Exception as e:
        logger.error(f"Enhanced sheet update failed: {e}")
        raise Exception(f"Enhanced sheet update failed: {str(e)}")

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

def build_decision_tree_from_calls(calls_data):
    """Build decision tree structure from multiple calls"""
    try:
        # Prepare aggregated decision data
        decision_summary = []
        for call in calls_data:
            summary = f"""
Call {call['row_number']} - {call['appointment_type']} / {call['appointment_reason']}:
Decision Logic: {call.get('decision_logic', 'Not available')}
Patient Type: {call.get('patient_type_determination', 'Not available')}
Symptoms: {call.get('symptom_assessment', 'Not available')}
Provider Logic: {call.get('provider_selection_logic', 'Not available')}
Workflow Summary: {call.get('summary', 'Not available')}
"""
            decision_summary.append(summary)
        
        tree_prompt = f"""
You are building a comprehensive decision tree for medical practice scheduling based on multiple call examples.

CALL DATA:
{chr(10).join(decision_summary)}

DECISION TREE CONSTRUCTION INSTRUCTIONS:
Analyze these calls to build a structured decision tree that shows:

1. **ROOT DECISION POINTS** - What are the first questions that determine the path?
2. **BRANCHING LOGIC** - How do patient responses create different pathways?
3. **PROVIDER ROUTING** - What rules determine provider assignment?
4. **APPOINTMENT TYPE MAPPING** - How do symptoms/needs map to appointment types?

Create a decision tree structure using this format:

**DECISION_TREE_STRUCTURE:**

```
START
├── New Patient?
│   ├── YES → Collect Basic Demographics
│   │   └── What brings you in today?
│   │       ├── Eye Pain/Emergency → Route to Emergency Protocol
│   │       ├── Routine Eye Exam → Schedule with General Ophthalmologist
│   │       ├── Specific Condition → Route to Specialist
│   │       └── Cosmetic Concerns → Route to Cosmetic Specialist
│   └── NO (Existing Patient) → Verify Identity
│       └── Reason for Visit?
│           ├── Follow-up → Check with Previous Provider
│           └── New Issue → Assess Symptoms
```

**DECISION_RULES:**
[List the specific rules that govern routing decisions]

**PROVIDER_SPECIALIZATIONS:**
[Map which providers handle which conditions/appointment types]

**APPOINTMENT_TYPE_MATRIX:**
[Show how patient needs map to appointment types]

**EXCEPTION_HANDLING:**
[How are special cases, emergencies, or complex cases handled?]

Focus on creating a reusable decision tree that new staff could follow.
"""
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a healthcare operations consultant specializing in creating standardized decision trees for patient scheduling workflows."},
                {"role": "user", "content": tree_prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        logger.info(f"Successfully built decision tree from {len(calls_data)} calls")
        return result
        
    except Exception as e:
        logger.error(f"Decision tree building failed: {e}")
        raise Exception(f"Decision tree building failed: {str(e)}")

def save_pattern_analysis(sheet_id, pattern_analysis, calls_analyzed):
    """Save pattern analysis to a separate sheet"""
    try:
        creds = get_service_account_credentials(SHEETS_SCOPES)
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
        timestamp = str(datetime.now())
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

def save_decision_tree_analysis(sheet_id, decision_tree, calls_analyzed):
    """Save decision tree analysis to a separate sheet"""
    try:
        creds = get_service_account_credentials(SHEETS_SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        
        # Create Decision Tree Analysis sheet if it doesn't exist
        try:
            sheet_metadata = service.spreadsheets().get(spreadsheetId=sheet_id).execute()
            sheets = sheet_metadata.get('sheets', [])
            tree_sheet_exists = any(sheet['properties']['title'] == 'Decision Tree Analysis' for sheet in sheets)
            
            if not tree_sheet_exists:
                request = {
                    'addSheet': {
                        'properties': {
                            'title': 'Decision Tree Analysis'
                        }
                    }
                }
                service.spreadsheets().batchUpdate(
                    spreadsheetId=sheet_id,
                    body={'requests': [request]}
                ).execute()
                logger.info("Created 'Decision Tree Analysis' sheet")
        
        except Exception as e:
            logger.warning(f"Could not create Decision Tree Analysis sheet: {e}")
        
        # Prepare the data
        timestamp = str(datetime.now())
        header_row = [f"Decision Tree Analysis - {timestamp}", f"Based on calls: {', '.join(map(str, calls_analyzed))}"]
        tree_rows = decision_tree.split('\n')
        
        # Insert the data
        values = [header_row, []] + [[row] for row in tree_rows]
        
        service.spreadsheets().values().append(
            spreadsheetId=sheet_id,
            range="Decision Tree Analysis!A:A",
            valueInputOption="RAW",
            insertDataOption="INSERT_ROWS",
            body={"values": values}
        ).execute()
        
        logger.info("Successfully saved decision tree analysis")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save decision tree analysis: {e}")
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
    return "AGEI Call Analysis Webhook Service - Enhanced with Decision Tree Analysis"

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
        
        # Enhanced analysis with decision logic
        enhanced_analysis = analyze_scheduling_workflow_enhanced(full_transcript, patient_context)
        workflow_fields, decision_fields = parse_enhanced_analysis(enhanced_analysis)
        
        # Save enhanced transcript file
        txt_path = mp3_path.replace(".mp3", "_enhanced_transcript.txt")
        with open(txt_path, "w", encoding='utf-8') as f:
            f.write("=== FULL CONVERSATION TRANSCRIPT ===\n\n")
            f.write(full_transcript)
            f.write("\n\n=== ENHANCED WORKFLOW & DECISION ANALYSIS ===\n\n")
            f.write(enhanced_analysis)
        
        # Upload to Drive and update sheet with enhanced data
        transcript_link = upload_to_drive(txt_path, data["transcript_folder_id"])
        update_sheet_enhanced(data["sheet_id"], data["sheet_name"], data["row_number"], 
                             transcript_link, workflow_fields, decision_fields)
        
        logger.info(f"Successfully processed call: {data['file_name']}")
        return jsonify({
            "status": "success", 
            "transcript_link": transcript_link,
            "workflow_fields": workflow_fields,
            "decision_fields": decision_fields
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing call: {error_msg}")
        
        # Try to update sheet with error status
        try:
            if 'data' in locals() and data:
                creds = get_service_account_credentials(SHEETS_SCOPES)
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

@app.route('/analyze-decision-tree', methods=['POST'])
def analyze_decision_tree_endpoint():
    """Endpoint for building decision trees from call data"""
    try:
        data = request.json
        
        if not data or 'calls_data' not in data:
            raise Exception("No calls data provided")
        
        calls_data = data['calls_data']
        sheet_id = data.get('sheet_id')
        
        logger.info(f"Building decision tree from {len(calls_data)} calls")
        
        # Build decision tree
        decision_tree = build_decision_tree_from_calls(calls_data)
        
        # Save to sheet
        calls_analyzed = [call['row_number'] for call in calls_data]
        save_decision_tree_analysis(sheet_id, decision_tree, calls_analyzed)
        
        logger.info("Decision tree analysis completed successfully")
        return jsonify({
            "status": "success",
            "calls_analyzed": len(calls_data),
            "decision_tree": decision_tree
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in decision tree analysis: {error_msg}")
        return jsonify({"error": error_msg}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)

# === FAQ Processing Enhancement - Add to your main.py ===

def analyze_faq_content(transcript):
    """Extract FAQ-worthy content from call transcript using AGEI-specific prompt"""
    try:
        faq_prompt = f"""
Identity: You are a careful and detail-oriented medical office analyst for Assil Gaur Eye Institute (AGEI). Your task is to review transcripts of calls between patients and the AGEI scheduling/reception team.

Goal: Your goal is to identify any information the agent communicated that could/should be part of a Frequently Asked Questions (FAQ) list for patients. Examples include:
- Parking instructions or directions
- When to arrive before an appointment
- What forms, IDs, insurance cards, or documents to bring
- Policies about late arrivals, cancellations, or rescheduling
- Payment or insurance coverage reminders
- Preparation instructions (e.g., "don't wear contacts before exam")
- Follow-up visit expectations

Do not include general chit-chat or anything that is not actionable or informational for patients.

Instructions: Read the transcript carefully. Extract agent-provided details that would be useful for other patients if added to a FAQ. Summarize each extracted detail clearly and concisely, in patient-friendly language. If no FAQ-worthy details were given, explicitly state: "No FAQ items mentioned in this call."

CALL TRANSCRIPT:
{transcript}

Output Format: Respond in the following JSON structure:
{{
  "FAQ_Items": [
    {{
      "Topic": "Parking validation at Santa Monica office",
      "FAQ_Text": "You can park in the structure next to the Santa Monica office; bring your ticket for validation."
    }},
    {{
      "Topic": "Arrival time for appointments", 
      "FAQ_Text": "Please arrive 15 minutes before your scheduled appointment to complete paperwork."
    }}
  ],
  "Notes": "Summaries are based only on what the agent explicitly said in this transcript."
}}
"""
        
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a medical office analyst for AGEI. Extract actionable patient information for FAQ creation. Always respond with valid JSON format."},
                {"role": "user", "content": faq_prompt}
            ],
            temperature=0.1
        )
        
        result = response.choices[0].message.content
        logger.info("Successfully analyzed FAQ content with AGEI prompt")
        return result
        
    except Exception as e:
        logger.error(f"FAQ analysis failed: {e}")
        raise Exception(f"FAQ analysis failed: {str(e)}")

def parse_faq_analysis(analysis_text):
    """Parse the JSON FAQ analysis response"""
    try:
        # Try to extract JSON from the response
        json_match = re.search(r'\{.*\}', analysis_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            faq_data = json.loads(json_str)
            
            # Extract FAQ items and format for sheet
            faq_items = faq_data.get('FAQ_Items', [])
            notes = faq_data.get('Notes', '')
            
            # Format FAQ items for display
            formatted_items = []
            for item in faq_items:
                topic = item.get('Topic', '')
                text = item.get('FAQ_Text', '')
                if topic and text:
                    formatted_items.append(f"• {topic}: {text}")
            
            return {
                "FAQ_ITEMS": "\n".join(formatted_items) if formatted_items else "No FAQ items mentioned in this call.",
                "TOTAL_ITEMS": str(len(formatted_items)),
                "NOTES": notes,
                "RAW_JSON": json_str  # Keep raw JSON for potential future use
            }
        else:
            logger.warning("No JSON found in FAQ analysis response")
            return {
                "FAQ_ITEMS": "No FAQ items mentioned in this call.",
                "TOTAL_ITEMS": "0",
                "NOTES": "Could not parse response",
                "RAW_JSON": analysis_text
            }
            
    except json.JSONDecodeError as e:
        logger.error(f"JSON parsing failed: {e}")
        return {
            "FAQ_ITEMS": "Error parsing FAQ analysis",
            "TOTAL_ITEMS": "0", 
            "NOTES": f"JSON parsing error: {str(e)}",
            "RAW_JSON": analysis_text
        }
    except Exception as e:
        logger.error(f"FAQ parsing failed: {e}")
        return {
            "FAQ_ITEMS": "Error processing FAQ analysis",
            "TOTAL_ITEMS": "0",
            "NOTES": f"Processing error: {str(e)}",
            "RAW_JSON": analysis_text
        }

def update_faq_sheet(sheet_id, sheet_name, row_number, faq_fields):
    """Update the FAQ sheet with analysis results"""
    try:
        creds = get_service_account_credentials(SHEETS_SCOPES)
        service = build('sheets', 'v4', credentials=creds)
        
        # Prepare FAQ values (columns B through E)
        faq_values = [
            faq_fields["FAQ_ITEMS"],      # B: FAQ Items Found
            faq_fields["TOTAL_ITEMS"],    # C: Number of Items
            faq_fields["NOTES"],          # D: Analysis Notes
            "Completed"                   # E: Processing Status
        ]
        
        # Update range B to E for the specific row
        update_range = f"{sheet_name}!B{row_number}:E{row_number}"
        
        service.spreadsheets().values().update(
            spreadsheetId=sheet_id,
            range=update_range,
            valueInputOption="RAW",
            body={"values": [faq_values]}
        ).execute()
        
        logger.info(f"Updated FAQ sheet row {row_number} successfully")
        return True
        
    except Exception as e:
        logger.error(f"FAQ sheet update failed: {e}")
        raise Exception(f"FAQ sheet update failed: {str(e)}")

# Add new FAQ endpoint
@app.route('/analyze-faq', methods=['POST'])
def analyze_faq():
    """Endpoint specifically for FAQ processing"""
    # Initialize file paths for cleanup
    mp3_path = None
    left_path = None
    right_path = None
    txt_path = None
    
    try:
        data = request.json
        
        # Handle test requests
        if data and data.get('test') == True:
            logger.info("FAQ test request received")
            return jsonify({"status": "success", "message": "FAQ webhook is working"})
        
        logger.info(f"Processing FAQ call: {data.get('file_name', 'unknown') if data else 'No data'}")
        
        # Validate required fields
        if not data:
            raise Exception("No JSON data received")
            
        required_fields = ['file_url', 'file_token', 'sheet_id', 'sheet_name', 
                          'row_number', 'transcript_folder_id', 'file_name']
        for field in required_fields:
            if field not in data:
                raise Exception(f"Missing required field: {field}")
        
        # Download and process audio (same as regular processing)
        mp3_path = download_file(data['file_url'], data['file_token'])
        left_path, right_path = split_channels(mp3_path)
        
        # Transcribe both channels
        agent_text = transcribe(left_path)
        patient_text = transcribe(right_path)
        
        # Reconstruct conversation
        full_transcript = reconstruct_conversation(agent_text, patient_text)
        
        # FAQ-specific analysis
        faq_analysis = analyze_faq_content(full_transcript)
        faq_fields = parse_faq_analysis(faq_analysis)
        
        # Save FAQ transcript file
        txt_path = mp3_path.replace(".mp3", "_faq_transcript.txt")
        with open(txt_path, "w", encoding='utf-8') as f:
            f.write("=== FULL CONVERSATION TRANSCRIPT ===\n\n")
            f.write(full_transcript)
            f.write("\n\n=== FAQ CONTENT ANALYSIS ===\n\n")
            f.write(faq_analysis)
        
        # Upload to Drive and update FAQ sheet
        transcript_link = upload_to_drive(txt_path, data["transcript_folder_id"])
        update_faq_sheet(data["sheet_id"], data["sheet_name"], data["row_number"], faq_fields)
        
        logger.info(f"Successfully processed FAQ call: {data['file_name']}")
        return jsonify({
            "status": "success", 
            "transcript_link": transcript_link,
            "faq_fields": faq_fields
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing FAQ call: {error_msg}")
        
        # Try to update sheet with error status
        try:
            if 'data' in locals() and data:
                creds = get_service_account_credentials(SHEETS_SCOPES)
                service = build('sheets', 'v4', credentials=creds)
                
                error_range = f"{data['sheet_name']}!F{data['row_number']}"
                service.spreadsheets().values().update(
                    spreadsheetId=data["sheet_id"],
                    range=error_range,
                    valueInputOption="RAW",
                    body={"values": [[f"ERROR: {error_msg}"]]}
                ).execute()
        except:
            logger.error("Could not update FAQ sheet with error status")
        
        return jsonify({"error": error_msg}), 500
        
    finally:
        # Clean up all temporary files
        cleanup_files(mp3_path, left_path, right_path, txt_path)
