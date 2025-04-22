import streamlit as st
import pandas as pd
from utils import parse_resume, get_job_matches, get_resume_tips, get_career_agent_prompt
from utils import generate_improved_resume, get_ai_response, extract_text_from_file
from utils import GenAI, check_openai_key
import time
from config import load_openai_api_key
import os

# Set page configuration
st.set_page_config(
    page_title="NextSteps - AI Career Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load OpenAI API key
openai_api_key = load_openai_api_key()
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Initialize session state variables if they don't exist
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
    
# Define navigation function
def navigate_to(page):
    st.session_state['current_page'] = page

# Sidebar navigation
with st.sidebar:
    st.title("NextSteps 🚀")
    st.subheader("AI Career Assistant")
    
    # Verify API key
    if not openai_api_key:
        st.error("⚠️ Please provide your OpenAI API key to use all features!")
    elif not check_openai_key():
        st.warning("⚠️ Your OpenAI API key may be invalid. Some features might not work.")
    
    # Only show navigation options if resume is uploaded
    if st.session_state['parsed_resume'] is not None:
        st.write("Navigate:")
        if st.button("🏠 Home", key="nav_home"):
            navigate_to('Home')
        if st.button("🔍 Job Matches", key="nav_jobs"):
            navigate_to('Job Matches')
        if st.button("📝 Resume Feedback", key="nav_feedback"):
            navigate_to('Resume Feedback')
        if st.button("💬 Career Agent", key="nav_agent"):
            navigate_to('Career Agent')
    else:
        st.info("Upload your resume to unlock all features!")
    
    st.divider()
    st.caption("NextSteps - Your AI Career Partner")

# Home Page
def show_home_page():
    st.title("Welcome to NextSteps")
    st.subheader("Your AI-Powered Career Assistant")
    
    # Check if API key is provided
    if not openai_api_key:
        st.warning("⚠️ Please provide your OpenAI API key in the sidebar to use all features of NextSteps.")
        st.info("Your API key is stored only in your local session and is not saved by the application.")
        st.session_state['api_key_warning_shown'] = True
        return
    
    if st.session_state['parsed_resume'] is None:
        st.write("Upload your resume (PDF or DOCX) to get started:")
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'docx'], key="resume_uploader")
        
        if uploaded_file is not None:
            with st.status("Processing your resume...") as status:
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
                st.session_state['job_matches'] = get_job_matches(st.session_state['parsed_resume'])
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
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 View Job Matches", key="home_to_matches"):
                navigate_to('Job Matches')
            if st.button("💬 Ask Career Agent", key="home_to_agent"):
                navigate_to('Career Agent')
        with col2:
            if st.button("📝 Get Resume Tips", key="home_to_feedback"):
                navigate_to('Resume Feedback')
            if st.button("🔄 Upload New Resume", key="upload_new"):
                for key in ['parsed_resume', 'job_matches', 'resume_tips', 'persona_prompt', 'messages', 'resume_text']:
                    if key in st.session_state:
                        st.session_state[key] = None
                st.rerun()

# Job Matches Page
def show_job_matches():
    st.title("Job Matches")
    
    if st.session_state['job_matches'] is None or not isinstance(st.session_state['job_matches'], pd.DataFrame):
        st.warning("No job matches found. Please upload your resume first.")
        if st.button("Return to Home"):
            navigate_to('Home')
    else:
        # Filter for minimum match score
        min_score = st.slider("Minimum Match Score (%)", 0, 100, 50, key="min_score")
        
        # Filter the dataframe
        filtered_matches = st.session_state['job_matches'][
            st.session_state['job_matches']['Match Score (%)'] >= min_score
        ]
        
        if filtered_matches.empty:
            st.info(f"No jobs match your minimum score of {min_score}%. Try lowering the threshold.")
        else:
            st.write(f"Found {len(filtered_matches)} matching jobs for you:")
            
            # Display the job matches
            st.dataframe(
                filtered_matches,
                column_config={
                    "Title": st.column_config.TextColumn("✅ Title"),
                    "Company": st.column_config.TextColumn("🏢 Company"),
                    "Match Score (%)": st.column_config.ProgressColumn("🔢 Match Score (%)", format="%d%%", min_value=0, max_value=100)
                },
                hide_index=True,
                use_container_width=True
            )
            
            # Allow user to select a job for more details
            selected_indices = st.multiselect(
                "Select a job to view more details:",
                options=filtered_matches.index,
                format_func=lambda x: f"{filtered_matches.loc[x, 'Title']} at {filtered_matches.loc[x, 'Company']}"
            )
            
            if selected_indices:
                for idx in selected_indices:
                    job = filtered_matches.loc[idx]
                    with st.expander(f"{job['Title']} at {job['Company']} - {job['Match Score (%)']}% Match", expanded=True):
                        st.markdown(f"**Job Description:**")
                        if 'Description' in job:
                            st.write(job['Description'])
                        else:
                            st.write("Detailed job description not available.")
                        
                        st.markdown(f"**Location:** {job.get('Location', 'Not specified')}")
                        st.markdown(f"**Salary Range:** {job.get('Salary Range', 'Not specified')}")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button("Tailor Resume For This Job", key=f"tailor_{idx}"):
                                st.session_state['selected_job'] = job
                                navigate_to('Resume Feedback')
                        with col2:
                            if st.button("Ask Career Agent About This Job", key=f"ask_agent_{idx}"):
                                # Pre-populate a question about this job for the career agent
                                job_question = f"How should I approach applying for this {job['Title']} position at {job['Company']}? What should I emphasize in my application?"
                                st.session_state['messages'].append({'role': 'user', 'content': job_question})
                                navigate_to('Career Agent')

# Resume Feedback Page
def show_resume_feedback():
    st.title("Resume Feedback")
    
    if st.session_state['parsed_resume'] is None or st.session_state['resume_tips'] is None:
        st.warning("No resume analysis available. Please upload your resume first.")
        if st.button("Return to Home"):
            navigate_to('Home')
    else:
        # Check if we're analyzing for a specific job
        specific_job = 'selected_job' in st.session_state and st.session_state['selected_job'] is not None
        
        if specific_job:
            job = st.session_state['selected_job']
            st.subheader(f"Resume Analysis for: {job['Title']} at {job['Company']}")
            st.progress(job['Match Score (%)'] / 100)
            st.write(f"Your resume has a {job['Match Score (%)']}% match with this job.")
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
                            for line in exp['description'].split('\n'):
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
                target_job = st.session_state.get('selected_job', None)
                improved_resume = generate_improved_resume(
                    st.session_state['parsed_resume'], 
                    target_job
                )
                time.sleep(1)  # Small delay for UI
            
            st.success("✅ Enhanced resume generated!")
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
        
        # Clear selected job when leaving page
        if specific_job and st.button("Back to Job Matches"):
            st.session_state.pop('selected_job', None)
            navigate_to('Job Matches')

# Career Agent Page
def show_career_agent():
    st.title("Career Agent Chat")
    
    if st.session_state['parsed_resume'] is None or st.session_state['persona_prompt'] is None:
        st.warning("Career agent not available. Please upload your resume first.")
        if st.button("Return to Home"):
            navigate_to('Home')
    else:
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

# Main app logic - determine which page to show
if st.session_state['current_page'] == 'Home':
    show_home_page()
elif st.session_state['current_page'] == 'Job Matches':
    show_job_matches()
elif st.session_state['current_page'] == 'Resume Feedback':
    show_resume_feedback()
elif st.session_state['current_page'] == 'Career Agent':
    show_career_agent()