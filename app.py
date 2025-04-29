import os
import streamlit as st
import pandas as pd
import pdfplumber
import docx2txt
import time
import json
import openai
import re
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="NextSteps - AI Career Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load and validate OpenAI API key
def load_openai_api_key():
    """Load OpenAI API key from environment or get it from user"""
    key = os.getenv("OPENAI_API_KEY")
    return key

def check_openai_key():
    """Validate OpenAI API key"""
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        client.models.list()
        return True
    except Exception:
        return False

# Text extraction functions
def extract_text_from_pdf(file):
    """Extract text from PDF using pdfplumber"""
    try:
        with pdfplumber.open(file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

def extract_text_from_docx(file):
    """Extract text from DOCX file"""
    try:
        text = docx2txt.process(file)
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_file(file):
    """Extract text from uploaded file based on extension"""
    if file is None:
        return ""
    
    file_content = file.getvalue()
    file_type = file.name.split('.')[-1].lower()
    
    if file_type == 'pdf':
        file_stream = BytesIO(file_content)
        return extract_text_from_pdf(file_stream)
    elif file_type in ['docx', 'doc']:
        file_stream = BytesIO(file_content)
        return extract_text_from_docx(file_stream)
    else:
        st.error(f"Unsupported file type: {file_type}")
        return ""

# AI and Resume parsing functions
def get_ai_response(prompt, messages, model="gpt-4"):
    """Get response from OpenAI API"""
    try:
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        
        # Create system prompt
        system_message = {"role": "system", "content": prompt}
        
        # Format conversation history
        formatted_messages = [system_message]
        for msg in messages:
            formatted_messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Call API
        response = client.chat.completions.create(
            model=model,
            messages=formatted_messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error calling OpenAI API: {e}")
        return "I'm sorry, I encountered an error while processing your request."

def parse_resume(uploaded_file):
    """Parse resume text into structured data using OpenAI"""
    try:
        resume_text = extract_text_from_file(uploaded_file)
        if not resume_text:
            return None
        
        # Create prompt for resume parsing
        prompt = f"""
        Parse the following resume text into a structured JSON format with these sections:
        - contact (name, email, phone, location)
        - summary (a brief professional summary)
        - skills (array of skill keywords)
        - experience (array of jobs with title, company, start_date, end_date, description)
        - education (array of education with degree, institution, start_date, end_date)
        
        If a section is not found, include an empty array or object for that section.
        For dates, use format 'YYYY-MM' when possible, or 'Present' for current positions.
        
        Resume text:
        {resume_text}
        
        Return only the JSON with no explanations or additional text.
        """
        
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.2,
            max_tokens=2000
        )
        
        # Extract and clean the JSON response
        json_text = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        json_text = re.sub(r'^```json\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text)
        
        parsed_data = json.loads(json_text)
        return parsed_data
    except Exception as e:
        st.error(f"Error parsing resume: {e}")
        return None

def get_job_matches(parsed_resume, soc_data, min_score=50):
    """Match resume to SOC jobs and filter by min_score"""
    if not parsed_resume:
        return pd.DataFrame()
    
    keywords = []
    
    # Extract keywords from resume
    if 'skills' in parsed_resume and parsed_resume['skills']:
        keywords += parsed_resume['skills']
    if 'experience' in parsed_resume and parsed_resume['experience']:
        for exp in parsed_resume['experience']:
            if isinstance(exp, dict):
                keywords.append(exp.get('title', ''))
                if 'description' in exp and exp['description']:
                    keywords += exp['description'].split()
    if 'summary' in parsed_resume and parsed_resume['summary']:
        keywords += parsed_resume['summary'].split()
    
    # Normalize keywords
    keywords = list(set([kw.lower() for kw in keywords if kw and len(kw) > 2]))
    
    # Score match
    soc_data['Match Score'] = soc_data.apply(lambda row: sum(
        kw in str(row['SOC Title']).lower() or kw in str(row['SOC Definition']).lower()
        for kw in keywords
    ), axis=1)
    
    matched = soc_data[soc_data['Match Score'] > 0].copy()
    if matched.empty:
        return pd.DataFrame()
        
    matched['Match Score (%)'] = (matched['Match Score'] / matched['Match Score'].max() * 100).round(1)
    matched = matched.sort_values(by='Match Score (%)', ascending=False)
    
    # Filter by minimum score
    matched = matched[matched['Match Score (%)'] >= min_score]
    
    # Rename columns to match expected display
    matched = matched.rename(columns={
        'SOC Title': 'Title',
        'SOC Definition': 'Description'
    })
    
    # Return only expected columns
    return matched[['SOC Code', 'Title', 'Description', 'Match Score (%)']]

def find_real_jobs(parsed_resume, job_title=None, location=None):
    """Find real job listings using OpenAI API"""
    try:
        # Prepare resume summary
        skills = ", ".join(parsed_resume.get('skills', [])[:10])
        experience_years = len(parsed_resume.get('experience', []))
        
        if job_title:
            search_job = job_title
        elif parsed_resume.get('experience') and len(parsed_resume['experience']) > 0:
            search_job = parsed_resume['experience'][0].get('title', 'Professional')
        else:
            search_job = "Professional"
        
        prompt = f"""
        Based on the following candidate profile, create 5 realistic job postings that would be a good match.
        
        Candidate Profile:
        - Skills: {skills}
        - Years of Experience: {experience_years}
        - Target Job Title: {search_job}
        - Preferred Location: {location if location else 'Any'}
        
        For each job posting, include:
        1. Job Title
        2. Company Name
        3. Location
        4. Salary Range
        5. Brief Job Description
        6. Required Skills
        7. Match Score (percentage)
        8. Match Reason (brief explanation of why this job is a good match)
        
        Return the results as a JSON array with objects containing the fields above.
        """
        
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        # Extract and clean the JSON response
        json_text = response.choices[0].message.content.strip()
        # Remove markdown code blocks if present
        json_text = re.sub(r'^```json\s*', '', json_text)
        json_text = re.sub(r'\s*```$', '', json_text)
        
        jobs = json.loads(json_text)
        return jobs
    except Exception as e:
        st.error(f"Error finding real jobs: {e}")
        return []

def get_resume_tips(parsed_resume):
    """Generate resume improvement tips using OpenAI"""
    if not parsed_resume:
        return []
    
    try:
        skills = ", ".join(parsed_resume.get('skills', [])[:10])
        experience_summary = ""
        for exp in parsed_resume.get('experience', [])[:2]:
            if isinstance(exp, dict):
                experience_summary += f"- {exp.get('title', 'Role')} at {exp.get('company', 'Company')}\n"
        
        prompt = f"""
        Analyze this resume summary and provide 5 specific, actionable tips to improve it:
        
        Skills: {skills}
        
        Experience:
        {experience_summary}
        
        Summary: {parsed_resume.get('summary', 'No summary provided')}
        
        Return only the numbered list of 5 tips with no additional text.
        """
        
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Parse the response
        tips_text = response.choices[0].message.content.strip()
        tips = [tip.strip() for tip in tips_text.split('\n') if tip.strip()]
        return tips
    except Exception as e:
        st.error(f"Error generating resume tips: {e}")
        return []

def get_career_agent_prompt(parsed_resume):
    """Generate persona prompt for career agent"""
    if not parsed_resume:
        return "You are a helpful career coach."
    
    skills = ", ".join(parsed_resume.get('skills', [])[:10])
    experience_summary = ""
    for exp in parsed_resume.get('experience', [])[:2]:
        if isinstance(exp, dict):
            experience_summary += f"- {exp.get('title', 'Role')} at {exp.get('company', 'Company')}\n"
    
    return f"""
    You are a professional career coach specializing in helping job seekers optimize their career search.
    
    Your client has the following profile:
    
    Skills: {skills}
    
    Experience:
    {experience_summary}
    
    Summary: {parsed_resume.get('summary', 'No summary provided')}
    
    Provide personalized, actionable advice tailored to this specific profile.
    Be concise but thorough, realistic but encouraging, and focus on practical steps.
    """

def generate_improved_resume(parsed_resume, target_job=None):
    """Generate improved resume content in DOCX format"""
    try:
        target_job_info = ""
        if target_job is not None:
            target_job_info = f"""
            Target Job Information:
            - Title: {target_job.get('Title', '')}
            - Description: {target_job.get('Description', '')}
            """
        
        prompt = f"""
        Improve this resume for maximum impact:
        
        {json.dumps(parsed_resume, indent=2)}
        
        {target_job_info}
        
        Focus on:
        1. Creating an impactful professional summary
        2. Enhancing bullet points with achievements and metrics
        3. Optimizing skills section
        4. Ensuring ATS-friendly formatting
        
        Return the improved content in a format suitable for a Word document.
        """
        
        client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        
        # For simplicity, we'll just return the response text
        # In a real implementation, you would convert this to a DOCX file
        improved_content = response.choices[0].message.content.strip()
        
        # Create a simple Word document from the content
        from docx import Document
        doc = Document()
        
        # Add the content to the document
        paragraphs = improved_content.split('\n\n')
        for p in paragraphs:
            if p.strip():
                doc.add_paragraph(p.strip())
        
        # Save to a BytesIO object
        output = BytesIO()
        doc.save(output)
        output.seek(0)
        
        return output.getvalue()
    except Exception as e:
        st.error(f"Error generating improved resume: {e}")
        return b""

# Load SOC 2018 job data
@st.cache_data
def load_soc_data():
    """Load SOC 2018 job data from CSV"""
    try:
        return pd.read_csv("data/soc_2018_definitions_clean.csv")
    except Exception as e:
        st.error(f"Error loading SOC data: {e}")
        # Create an empty dataframe with necessary columns
        return pd.DataFrame(columns=['SOC Code', 'SOC Title', 'SOC Definition'])

# Navigation function
def navigate_to(page):
    """Change the current page in session state"""
    st.session_state['current_page'] = page

# Initialize session state variables
def init_session_state():
    """Initialize all session state variables"""
    if 'parsed_resume' not in st.session_state:
        st.session_state['parsed_resume'] = None
    if 'job_matches' not in st.session_state:
        st.session_state['job_matches'] = None
    if 'resume_tips' not in st.session_state:
        st.session_state['resume_tips'] = None
    if 'persona_prompt' not in st.session_state:
        st.session_state['persona_prompt'] = None
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'current_page' not in st.session_state:
        st.session_state['current_page'] = 'Home'
    if 'resume_text' not in st.session_state:
        st.session_state['resume_text'] = None
    if 'api_key_warning_shown' not in st.session_state:
        st.session_state['api_key_warning_shown'] = False
    if 'real_jobs' not in st.session_state:
        st.session_state['real_jobs'] = None

# Main application components
def sidebar():
    """Create the sidebar navigation"""
    with st.sidebar:
        st.title("NextSteps 🚀")
        st.subheader("AI Career Assistant")
        
        # API Key input
        openai_api_key = st.text_input("Enter OpenAI API Key", type="password", key="api_key_input")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("API Key set!")
        elif not os.environ.get("OPENAI_API_KEY"):
            st.warning("⚠️ Please enter your OpenAI API key to use all features!")
        
        # Verify API key
        if os.environ.get("OPENAI_API_KEY") and not check_openai_key():
            st.warning("⚠️ Your OpenAI API key may be invalid. Some features might not work.")
        
        # Only show navigation options if resume is uploaded
        if st.session_state['parsed_resume'] is not None:
            st.write("Navigate:")
            if st.button("🏠 Home", key="nav_home"):
                navigate_to('Home')
            if st.button("🔍 Census Job Matches", key="nav_census_jobs"):
                navigate_to('Census Job Matches')
            if st.button("🔎 Real Job Matches", key="nav_real_jobs"):
                navigate_to('Real Job Matches')
            if st.button("📝 Resume Feedback", key="nav_feedback"):
                navigate_to('Resume Feedback')
            if st.button("💬 Career Agent", key="nav_agent"):
                navigate_to('Career Agent')
        else:
            st.info("Upload your resume to unlock all features!")
        
        st.divider()
        st.caption("NextSteps - Your AI Career Partner")

def home_page():
    """Display the home page"""
    st.title("Welcome to NextSteps")
    st.subheader("Your AI-Powered Career Assistant")
    
    # Check if API key is provided
    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("⚠️ Please provide your OpenAI API key in the sidebar to use all features of NextSteps.")
        st.info("Your API key is stored only in your local session and is not saved by the application.")
        st.session_state['api_key_warning_shown'] = True
        return
    
    if st.session_state['parsed_resume'] is None:
        st.write("Upload your resume (PDF or DOCX) to get started:")
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx'], key="resume_uploader")
        
        if uploaded_file is not None:
            with st.status("Processing your resume...") as status:
                # Load the SOC data
                soc_data = load_soc_data()
                if soc_data.empty:
                    st.error("Failed to load job data. Please check if the data file exists.")
                    return
                
                status.update(label="Extracting text from your resume...", state="running")
                # Extract and store the full text
                st.session_state['resume_text'] = extract_text_from_file(uploaded_file)
                if not st.session_state['resume_text']:
                    st.error("Failed to extract text from the uploaded file. Please try another file.")
                    return
                
                status.update(label="Parsing your resume with AI...", state="running")
                # Parse the resume
                st.session_state['parsed_resume'] = parse_resume(uploaded_file)
                time.sleep(0.5)  # Small delay for UI
                
                status.update(label="Finding job matches for your profile...", state="running")
                # Get job matches
                st.session_state['job_matches'] = get_job_matches(st.session_state['parsed_resume'], soc_data)
                time.sleep(0.5)
                
                status.update(label="Finding real job opportunities...", state="running")
                # Get real job listings
                st.session_state['real_jobs'] = find_real_jobs(st.session_state['parsed_resume'])
                time.sleep(0.5)
                
                status.update(label="Analyzing your resume for improvement tips...", state="running")
                # Get resume tips
                st.session_state['resume_tips'] = get_resume_tips(st.session_state['parsed_resume'])
                time.sleep(0.5)
                
                status.update(label="Preparing your career agent...", state="running")
                # Generate persona prompt for AI assistant
                st.session_state['persona_prompt'] = get_career_agent_prompt(st.session_state['parsed_resume'])
                time.sleep(0.5)
                
                status.update(label="Resume processed successfully!", state="complete")
                
            st.success("✅ Your resume has been analyzed successfully!")
            st.rerun()
    else:
        # Display resume summary
        st.subheader("Welcome! Here's what we learned about you...")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Professional Summary")
            if 'summary' in st.session_state['parsed_resume'] and st.session_state['parsed_resume']['summary']:
                st.write(st.session_state['parsed_resume']['summary'])
            else:
                st.caption("No summary extracted. Consider adding a professional summary to your resume.")
            
            st.markdown("### Top Skills")
            if 'skills' in st.session_state['parsed_resume'] and st.session_state['parsed_resume']['skills']:
                skills = st.session_state['parsed_resume']['skills']
                if isinstance(skills, list) and len(skills) > 0:
                    for skill in skills[:5]:  # Show top 5 skills
                        st.markdown(f"- {skill}")
                    if len(skills) > 5:
                        st.caption(f"... and {len(skills) - 5} more")
            else:
                st.caption("No skills extracted. Consider adding a skills section to your resume.")
        
        with col2:
            st.markdown("### Experience")
            if 'experience' in st.session_state['parsed_resume'] and st.session_state['parsed_resume']['experience']:
                experience = st.session_state['parsed_resume']['experience']
                if isinstance(experience, list) and len(experience) > 0:
                    for exp in experience[:2]:  # Show top 2 experiences
                        if isinstance(exp, dict):
                            st.markdown(f"**{exp.get('title', 'Role')}** at {exp.get('company', 'Company')}")
                            st.caption(f"{exp.get('start_date', '')} - {exp.get('end_date', 'Present')}")
                    if len(experience) > 2:
                        st.caption(f"... and {len(experience) - 2} more positions")
            else:
                st.caption("No experience extracted. Check that your work history is clearly formatted in your resume.")
            
            st.markdown("### Education")
            if 'education' in st.session_state['parsed_resume'] and st.session_state['parsed_resume']['education']:
                education = st.session_state['parsed_resume']['education']
                if isinstance(education, list) and len(education) > 0:
                    for edu in education[:1]:  # Show top education
                        if isinstance(edu, dict):
                            st.markdown(f"**{edu.get('degree', 'Degree')}** from {edu.get('institution', 'Institution')}")
                            if 'start_date' in edu or 'end_date' in edu:
                                st.caption(f"{edu.get('start_date', '')} - {edu.get('end_date', '')}")
            else:
                st.caption("No education extracted. Consider adding your educational background to your resume.")
        
        # Navigation buttons
        st.write("### What would you like to do next?")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 View Census Job Matches", key="home_to_census_matches"):
                navigate_to('Census Job Matches')
        with col2:
            if st.button("🔎 View Real Job Opportunities", key="home_to_real_jobs"):
                navigate_to('Real Job Matches')
        with col3:
            if st.button("💬 Ask Career Agent", key="home_to_agent"):
                navigate_to('Career Agent')
            
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Get Resume Tips", key="home_to_feedback"):
                navigate_to('Resume Feedback')
        with col2:
            if st.button("🔄 Upload New Resume", key="upload_new"):
                st.session_state['parsed_resume'] = None
                st.session_state['job_matches'] = None
                st.session_state['resume_tips'] = None
                st.session_state['persona_prompt'] = None
                st.session_state['messages'] = []  # ✅ set to empty list
                st.session_state['resume_text'] = None
                st.session_state['real_jobs'] = None
                navigate_to('Home')
                st.rerun()


def census_job_matches_page():
    """Display the Census job matches page"""
    st.title("Census Job Matches")
    
    if st.session_state['job_matches'] is None or not isinstance(st.session_state['job_matches'], pd.DataFrame):
        st.warning("No job matches found. Please upload your resume first.")
        if st.button("Return to Home"):
            navigate_to('Home')
    else:
        # Filter for minimum match score
        min_score = st.slider("Minimum Match Score (%)", 0, 100, 50, key="min_score")
        
        # Load the SOC data
        soc_data = load_soc_data()
        
        # Recompute job matches with new threshold
        filtered_matches = get_job_matches(st.session_state['parsed_resume'], soc_data, min_score)
        
        if filtered_matches.empty:
            st.info(f"No census jobs match your minimum score of {min_score}%. Try lowering the threshold.")
        else:
            st.write(f"Found {len(filtered_matches)} matching census jobs for you:")
            
            # Display the job matches
            st.dataframe(
                filtered_matches,
                column_config={
                    "Title": st.column_config.TextColumn("Job Title"),
                    "SOC Code": st.column_config.TextColumn("SOC Code"),
                    "Match Score (%)": st.column_config.ProgressColumn("Match Score (%)", format="%d%%", min_value=0, max_value=100)
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Allow user to select a job for more details
            selected_indices = st.multiselect(
                "Select a job to view more details:",
                options=filtered_matches.index,
                format_func=lambda x: filtered_matches.loc[x, 'Title']
            )
            
            if selected_indices:
                for idx in selected_indices:
                    job = filtered_matches.loc[idx]
                    with st.expander(f"{job['Title']} - {job['Match Score (%)']}% Match", expanded=True):
                        st.markdown(f"**SOC Code:** {job['SOC Code']}")
                        st.markdown(f"**Job Description:**")
                        if 'Description' in job:
                            st.write(job['Description'])
                        else:
                            st.write("Detailed job description not available.")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Tailor Resume For This Job", key=f"tailor_{idx}"):
                                st.session_state['selected_job'] = job
                                navigate_to('Resume Feedback')
                        with col2:
                            if st.button("Ask Career Agent About This Job", key=f"ask_agent_{idx}"):
                                # Pre-populate a question about this job for the career agent
                                job_question = f"How should I prepare for and apply to a {job['Title']} position? What skills should I emphasize?"
                                st.session_state['messages'].append({'role': 'user', 'content': job_question})
                                navigate_to('Career Agent')

def real_job_matches_page():
    """Display real job matches page"""
    st.title("Real Job Opportunities")
    
    if st.session_state['parsed_resume'] is None:
        st.warning("No resume uploaded. Please upload your resume first.")
        if st.button("Return to Home"):
            navigate_to('Home')
        return
        
    # Search options
    col1, col2 = st.columns(2)
    with col1:
        job_title = st.text_input("Job Title (optional)")
    with col2:
        location = st.text_input("Location (optional)")
    
    if st.button("Search Jobs") or st.session_state['real_jobs'] is None:
        with st.spinner("Finding job opportunities..."):
            st.session_state['real_jobs'] = find_real_jobs(
                st.session_state['parsed_resume'],
                job_title,
                location
            )
    
    # Display jobs
    if st.session_state['real_jobs']:
        st.write(f"Found {len(st.session_state['real_jobs'])} job opportunities matching your profile:")
        
        for i, job in enumerate(st.session_state['real_jobs']):
            with st.expander(f"{job['Job Title']} at {job['Company Name']} - {job['Match Score']}% Match", expanded=i==0):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**🏢 Company:** {job['Company Name']}")
                    st.markdown(f"**📍 Location:** {job['Location']}")
                    st.markdown(f"**💰 Salary Range:** {job['Salary Range']}")
                    st.markdown(f"**⭐ Match Score:** {job['Match Score']}%")
                
                with col2:
                    st.markdown("**🔍 Match Reason:**")
                    st.info(job['Match Reason'])
                    
                st.markdown("**📋 Job Description:**")
                st.write(job['Brief Job Description'])
                
                st.markdown("**🛠️ Required Skills:**")
                if isinstance(job['Required Skills'], list):
                    for skill in job['Required Skills']:
                        st.markdown(f"- {skill}")
                else:
                    st.write(job['Required Skills'])
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Tailor Resume For This Job", key=f"tailor_real_{i}"):
                        st.session_state['selected_real_job'] = job
                        navigate_to('Resume Feedback')
                with col2:
                    if st.button("Ask Career Agent About This Job", key=f"ask_agent_real_{i}"):
                        # Pre-populate a question about this job for the career agent
                        job_question = f"How should I prepare for and apply to this {job['Job Title']} position at {job['Company Name']}? What points should I emphasize in my application?"
                        st.session_state['messages'].append({'role': 'user', 'content': job_question})
                        navigate_to('Career Agent')

def resume_feedback_page():
    """Display resume feedback page"""
    st.title("Resume Feedback")
    
    if st.session_state['parsed_resume'] is None or st.session_state['resume_tips'] is None:
        st.warning("No resume analysis available. Please upload your resume first.")
        if st.button("Return to Home"):
            navigate_to('Home')
    else:
        # Check if we're analyzing for a specific job
        specific_job = False
        job_info = None
        
        if 'selected_job' in st.session_state and st.session_state['selected_job'] is not None:
            specific_job = True
            job_info = st.session_state['selected_job']
        elif 'selected_real_job' in st.session_state and st.session_state['selected_real_job'] is not None:
            specific_job = True
            job_info = st.session_state['selected_real_job']
        
        if specific_job and job_info is not None:
            if 'Title' in job_info:
                job_title = job_info['Title']
                job_company = job_info.get('Company', '')
                match_score = job_info.get('Match Score (%)', 0)
            else:
                job_title = job_info['Job Title']
                job_company = job_info['Company Name']
                match_score = job_info['Match Score']
                
            st.subheader(f"Resume Analysis for: {job_title} at {job_company}")
            st.progress(float(match_score) / 100)
            st.write(f"Your resume has a {match_score}% match with this job.")
        else:
            st.subheader("General Resume Analysis")
        
        # Resume summary
        with st.expander("📄 Your Resume Summary", expanded=not specific_job):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Professional Experience")
                if 'experience' in st.session_state['parsed_resume'] and st.session_state['parsed_resume']['experience']:
                    for exp in st.session_state['parsed_resume']['experience']:
                        st.markdown(f"**{exp.get('title', 'Role')}** at {exp.get('company', 'Company')}")
                        st.caption(f"{exp.get('start_date', '')} - {exp.get('end_date', 'Present')}")
                        if 'description' in exp and exp['description']:
                            description_lines = exp['description'].split('\n')
                            for line in description_lines:
                                if line.strip():
                                    st.markdown(f"- {line.strip()}")
                else:
                    st.caption("No experience extracted from your resume.")
            
            with col2:
                st.markdown("### Education")
                if 'education' in st.session_state['parsed_resume'] and st.session_state['parsed_resume']['education']:
                    for edu in st.session_state['parsed_resume']['education']:
                        st.markdown(f"**{edu.get('degree', 'Degree')}** from {edu.get('institution', 'Institution')}")
                        st.caption(f"{edu.get('start_date', '')} - {edu.get('end_date', '')}")
                else:
                    st.caption("No education extracted from your resume.")
                
                st.markdown("### Skills")
                if 'skills' in st.session_state['parsed_resume'] and st.session_state['parsed_resume']['skills']:
                    skills_text = ", ".join(st.session_state['parsed_resume']['skills'])
                    st.write(skills_text)
                else:
                    st.caption("No skills extracted from your resume.")
        
        # Resume tips
        st.subheader("💡 Improvement Suggestions")
        if isinstance(st.session_state['resume_tips'], list) and st.session_state['resume_tips']:
            for i, tip in enumerate(st.session_state['resume_tips']):
                st.markdown(f"**Tip {i+1}:** {tip}")
        else:
            st.info("No specific improvement tips available. Try uploading a more detailed resume.")
        
        # Generate improved resume option
        st.subheader("🚀 Enhance Your Resume")
        
        # Options for resume enhancement
        enhancement_options = st.multiselect(
            "Select enhancement options:",
            options=["Improve professional summary", "Enhance bullet points", "Reorganize skills", "Optimize for ATS"],
            default=["Improve professional summary", "Enhance bullet points", "Optimize for ATS"]
        )
        
        if st.button("Generate Improved Resume"):
            with st.spinner("Generating enhanced resume..."):
                target_job = None
                if specific_job:
                    target_job = job_info
                    
                improved_resume = generate_improved_resume(
                    st.session_state['parsed_resume'], 
                    target_job
                )
                time.sleep(1)  # Small delay for UI
            
            st.success("✅ Enhanced resume generated!")
            
            # Check if we have valid binary data
            if improved_resume and len(improved_resume) > 0:
                st.download_button(
                    label="Download Enhanced Resume (DOCX)",
                    data=improved_resume,
                    file_name="enhanced_resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                
                # Show what was improved
                st.markdown("### Enhancements Made:")
                if "Improve professional summary" in enhancement_options:
                    st.markdown("✅ **Professional summary** optimized for impact")
                if "Enhance bullet points" in enhancement_options:
                    st.markdown("✅ **Experience bullet points** enhanced with achievements and metrics")
                if "Reorganize skills" in enhancement_options:
                    st.markdown("✅ **Skills section** reorganized by category")
                if "Optimize for ATS" in enhancement_options:
                    st.markdown("✅ **ATS optimization** applied to improve matching with automated systems")
            else:
                st.error("Failed to generate enhanced resume. Please try again.")
        
        # Clear selected job when leaving page
        if specific_job:
            if st.button("Clear Selected Job"):
                if 'selected_job' in st.session_state:
                    st.session_state.pop('selected_job', None)
                if 'selected_real_job' in st.session_state:
                    st.session_state.pop('selected_real_job', None)
                st.rerun()

def career_agent_page():
    """Display career agent chat page"""
    st.title("Career Agent Chat")
    
    if st.session_state['parsed_resume'] is None or st.session_state['persona_prompt'] is None:
        st.warning("Career agent not available. Please upload your resume first.")
        if st.button("Return to Home"):
            navigate_to('Home')
    else:
        if st.session_state['messages'] is None:
            st.session_state['messages'] = []

        # Chat interface
        st.write("Chat with your AI career coach. Ask anything about your career, resume, or job search!")
        
        # Suggested questions
        with st.expander("Suggested questions to ask"):
            suggestions = [
                "How can I improve my resume for tech roles?",
                "What skills should I highlight for a Senior Developer position?",
                "How should I prepare for behavioral interviews?",
                "What salary range should I expect for my experience level?",
                "How can I negotiate a better compensation package?",
                "What are the best ways to showcase my technical projects?",
                "Should I include a cover letter with my application?"
            ]
            
            for suggestion in suggestions:
                if st.button(suggestion, key=f"suggest_{hash(suggestion)}"):
                    st.session_state['messages'].append({'role': 'user', 'content': suggestion})
                    st.rerun()
        
        # Display chat messages
        for message in st.session_state['messages']:
            if message['role'] == 'user':
                st.chat_message("user").write(message['content'])
            else:
                st.chat_message("assistant").write(message['content'])
        
        # Input for new message
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to chat history
            st.session_state['messages'].append({'role': 'user', 'content': user_input})
            
            # Display user message
            st.chat_message("user").write(user_input)
            
            # Get and display AI response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_ai_response(
                        st.session_state['persona_prompt'],
                        st.session_state['messages']
                    )
                
                st.write(response)
                
                # Add AI response to chat history
                st.session_state['messages'].append({'role': 'assistant', 'content': response})

# Main application entry point
def main():
    """Main application function"""
    # Initialize session state
    init_session_state()
    
    # Display sidebar
    sidebar()
    
    # Display appropriate page based on current_page
    if st.session_state['current_page'] == 'Home':
        home_page()
    elif st.session_state['current_page'] == 'Census Job Matches':
        census_job_matches_page()
    elif st.session_state['current_page'] == 'Real Job Matches':
        real_job_matches_page()
    elif st.session_state['current_page'] == 'Resume Feedback':
        resume_feedback_page()
    elif st.session_state['current_page'] == 'Career Agent':
        career_agent_page()

# Run the main application
if __name__ == "__main__":
    main()