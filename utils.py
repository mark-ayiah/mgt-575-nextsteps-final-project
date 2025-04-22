import pandas as pd
import io
import random
import PyPDF2
import docx
import re
import json
import base64
import os
import time
import requests
import openai
import traceback
from docx import Document

class GenAI:
    """
    A class for interacting with the OpenAI API to generate text, images, video descriptions,
    perform speech recognition, and handle basic document processing tasks.

    Attributes:
    ----------
    client : openai.Client
        An instance of the OpenAI client initialized with the API key.
    """
    def __init__(self, openai_api_key):
        """
        Initializes the GenAI class with the provided OpenAI API key.

        Parameters:
        ----------
        openai_api_key : str
            The API key for accessing OpenAI's services.
        """
        self.client = openai.Client(api_key=openai_api_key)
        self.openai_api_key = openai_api_key


    def generate_text(self, prompt, instructions='You are a helpful AI named Jarvis', model="gpt-4o-mini", response_format='text', temperature=1):
    # Convert string response_format to the expected object format
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
        response = completion.choices[0].message.content
        return response


    def generate_chat_response(self, chat_history, user_message, instructions, model="gpt-4o-mini", output_type='text'):
        """
        Generates a chatbot-like response based on the conversation history.

        Parameters:
        ----------
        chat_history : list
            List of previous messages, each as a dict with "role" and "content".
        user_message : str
            The latest message from the user.
        instructions : str
            System instructions defining the chatbot's behavior.
        model : str, optional
            The OpenAI model to use (default is 'gpt-4o-mini').
        output_type : str, optional
            The format of the output (default is 'text').

        Returns:
        -------
        str
            The chatbot's response.
        """
        # Add the latest user message to the chat history
        chat_history.append({"role": "user", "content": user_message})

        # Call the OpenAI API to get a response
        completion = self.client.chat.completions.create(
            model=model,
            response_format={"type": output_type},
            messages=[
                {"role": "system", "content": instructions},  # Add system instructions
                *chat_history  # Unpack the chat history to include all previous messages
            ]
        )

        # Extract the bot's response from the API completion
        bot_response = completion.choices[0].message.content

        # Add the bot's response to the chat history
        chat_history.append({"role": "assistant", "content": bot_response})

        return bot_response

# Resume Parsing Functions
def parse_resume(file):
    """
    Parse an uploaded resume file (PDF or DOCX) and extract structured information
    using OpenAI API for more accurate parsing.
    
    Args:
        file: The uploaded file object from Streamlit
    
    Returns:
        dict: A dictionary containing parsed resume sections
    """
    content = extract_text_from_file(file)
    
    # Create a GenAI instance with your OpenAI API key
    import os
    openai_api_key = os.environ.get("OPENAI_API_KEY", "your_api_key_here")
    gen_ai = GenAI(openai_api_key)
    
    # Prompt for GPT to parse the resume
    prompt = f"""
    Parse the following resume text into a structured JSON format with the following sections:
    - summary: A brief professional summary
    - experience: List of work experiences (title, company, start_date, end_date, description)
    - education: List of educational history (degree, institution, start_date, end_date, gpa)
    - skills: List of professional skills
    - contact: Contact information (name, email, phone, location)

    Resume text:
    {content}

    Return only the JSON object without any additional text.
    """
    
    # Get GPT's structured response
    try:
        response = gen_ai.generate_text(
            prompt=prompt,
            instructions="You are an expert resume parser. Extract structured information from resumes accurately.",
            model="gpt-4o-mini",
            response_format='json_object',
            temperature=0.1
        )
        
        # Parse the JSON response
        parsed = json.loads(response) if isinstance(response, str) else response
        
        # Ensure all expected sections exist
        expected_sections = ['summary', 'experience', 'education', 'skills', 'contact']
        for section in expected_sections:
            if section not in parsed:
                parsed[section] = [] if section in ['experience', 'education', 'skills'] else ""
        
        # Add raw text
        parsed['raw_text'] = content
        
        return parsed
    
    except Exception as e:
        # Fallback to traditional parsing if AI parsing fails
        print(f"GPT parsing failed: {e}. Falling back to traditional parsing.")
        parsed = {
            'raw_text': content,
            'summary': extract_summary(content),
            'experience': extract_experience(content),
            'education': extract_education(content),
            'skills': extract_skills(content),
            'contact': extract_contact_info(content)
        }
        
        return parsed

def extract_text_from_file(file):
    """Extract text content from PDF or DOCX file"""
    file_type = file.name.split('.')[-1].lower()
    file_content = file.read()
    
    if file_type == 'pdf':
        return extract_text_from_pdf(file_content)
    elif file_type == 'docx':
        return extract_text_from_docx(file_content)
    else:
        return "Unsupported file format"

def extract_text_from_pdf(file_content):
    """Extract text from PDF file"""
    try:
        pdf_file = io.BytesIO(file_content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        # For demo purposes, return dummy text if PDF extraction fails
        return """
        John Doe
        Software Engineer
        
        Experience:
        Senior Developer at Tech Corp (2019-Present)
        - Led team of 5 developers on cloud migration project
        - Implemented CI/CD pipeline reducing deployment time by 70%
        
        Junior Developer at StartUp Inc (2017-2019)
        - Developed mobile application with over 100,000 downloads
        - Created RESTful APIs for client-server communication
        
        Education:
        BS Computer Science, University of Technology (2013-2017)
        
        Skills:
        Python, JavaScript, React, Docker, AWS, Git, Agile, Machine Learning
        """

def extract_text_from_docx(file_content):
    """Extract text from DOCX file"""
    try:
        docx_file = io.BytesIO(file_content)
        doc = docx.Document(docx_file)
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        return text
    except Exception as e:
        # For demo purposes, return dummy text if DOCX extraction fails
        return """
        Jane Smith
        Data Scientist
        
        Experience:
        Lead Data Scientist at Analytics Co (2020-Present)
        - Developed customer segmentation model increasing conversion by 25%
        - Built recommendation engine used by over 2M users
        
        Data Analyst at Big Data Inc (2018-2020)
        - Created visualization dashboard for executive team
        - Performed A/B testing improving user engagement by 15%
        
        Education:
        MS Data Science, State University (2016-2018)
        BS Statistics, College University (2012-2016)
        
        Skills:
        Python, R, SQL, TensorFlow, PyTorch, Data Visualization, Statistical Analysis
        """

def extract_summary(text):
    """Extract professional summary from resume text"""
    # In a real implementation, this would use ML/NLP to identify the summary
    # For demo purposes, we'll return the first paragraph after removing contact info
    lines = text.split('\n')
    filtered_lines = [line for line in lines if line.strip() and not re.search(r'@|phone|email|\d{3}-\d{3}-\d{4}', line.lower())]
    
    # Get the first 2-3 non-empty lines after the name as the "summary"
    summary_lines = filtered_lines[1:4]  # Skip the first line (usually the name)
    summary = ' '.join(summary_lines)
    
    if len(summary) < 10:
        # Fallback generic summary if extraction fails
        return "Experienced professional with a track record of success in delivering results."
    
    return summary

def extract_experience(text):
    """Extract work experience from resume text"""
    # In a real implementation, this would use ML/NLP to identify and structure work experience
    # For demo purposes, we'll generate synthetic experience
    
    # Try to extract company names and titles from text
    companies = set(re.findall(r'at ([A-Z][A-Za-z\s]+)', text))
    titles = set(re.findall(r'(Developer|Engineer|Scientist|Analyst|Manager|Director)\b', text))
    
    # If we couldn't extract enough information, use default values
    if len(companies) < 2:
        companies = ["TechCorp", "StartUp Inc", "Innovative Solutions"] 
    if len(titles) < 2:
        titles = ["Senior Developer", "Software Engineer", "Project Lead"]
    
    # Create synthetic experience entries
    experience = []
    current_year = 2025
    
    for i in range(min(len(companies), 3)):  # Up to 3 experiences
        company = list(companies)[i] if i < len(companies) else f"Company {i+1}"
        title = list(titles)[i % len(titles)] if titles else f"Position {i+1}"
        
        end_year = current_year - (i * 2)
        start_year = end_year - random.randint(1, 3)
        
        experience.append({
            'title': title,
            'company': company,
            'start_date': str(start_year),
            'end_date': "Present" if i == 0 else str(end_year),
            'description': generate_experience_description(title)
        })
    
    return experience

def generate_experience_description(title):
    """Generate synthetic job description based on job title"""
    descriptions = {
        'Developer': [
            "Developed and maintained web applications using React and Node.js.",
            "Implemented RESTful APIs and improved system performance by 40%.",
            "Collaborated with cross-functional teams to deliver products on time."
        ],
        'Engineer': [
            "Designed and implemented scalable software architecture.",
            "Reduced system downtime by 30% through optimized database queries.",
            "Mentored junior engineers and led code reviews."
        ],
        'Scientist': [
            "Built machine learning models to predict customer behavior.",
            "Analyzed large datasets to extract actionable business insights.",
            "Created data visualization dashboards for stakeholders."
        ],
        'Analyst': [
            "Conducted A/B testing to optimize user experience.",
            "Generated monthly reports on key performance indicators.",
            "Identified trends in customer data that increased retention by 15%."
        ],
        'Manager': [
            "Led a team of 8 developers to deliver projects on schedule.",
            "Implemented agile methodology improving team productivity by 25%.",
            "Managed project budgets exceeding $1M annually."
        ],
        'Director': [
            "Oversaw department strategy and execution of company initiatives.",
            "Grew team from 5 to 20 while maintaining high-quality standards.",
            "Presented quarterly results to C-level executives."
        ]
    }
    
    # Find the most relevant key in the descriptions dictionary
    for key in descriptions:
        if key.lower() in title.lower():
            return "\n".join(descriptions[key])
    
    # Default description if no specific match
    default = [
        "Led key projects delivering business value.",
        "Collaborated with stakeholders to meet objectives.",
        "Improved processes and implemented best practices."
    ]
    return "\n".join(default)

def extract_education(text):
    """Extract education information from resume text"""
    # In a real implementation, this would use ML/NLP to identify education
    # For demo purposes, we'll generate synthetic education
    
    # Try to extract education info from text
    degrees = re.findall(r'(BS|MS|BA|MA|PhD|Bachelor|Master|Doctor)\s+(?:of|in)?\s+([A-Za-z\s]+)', text)
    universities = re.findall(r'(University|College|Institute|School)\s+of\s+([A-Za-z\s]+)', text)
    
    education = []
    
    if degrees and universities:
        for i in range(min(len(degrees), len(universities), 2)):  # Up to 2 education entries
            degree = ' '.join(degrees[i])
            university = ' '.join(universities[i])
            
            end_year = 2023 - (i * 4)
            start_year = end_year - random.randint(3, 5)
            
            education.append({
                'degree': degree,
                'institution': university,
                'start_date': str(start_year),
                'end_date': str(end_year),
                'gpa': f"{random.uniform(3.0, 4.0):.2f}"
            })
    else:
        # Default education if extraction fails
        education = [
            {
                'degree': 'BS Computer Science',
                'institution': 'University of Technology',
                'start_date': '2015',
                'end_date': '2019',
                'gpa': '3.8'
            }
        ]
    
    return education

def extract_skills(text):
    """Extract skills from resume text"""
    # Common skills to look for
    common_skills = [
        "Python", "JavaScript", "Java", "C++", "C#", "Ruby", "PHP", "Swift",
        "React", "Angular", "Vue", "Node.js", "Django", "Flask", "Spring", "ASP.NET",
        "SQL", "MongoDB", "PostgreSQL", "MySQL", "Oracle", "NoSQL",
        "AWS", "Azure", "GCP", "Docker", "Kubernetes", "CI/CD", "Jenkins",
        "Git", "SVN", "Agile", "Scrum", "Kanban", "Jira",
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Science",
        "TensorFlow", "PyTorch", "scikit-learn", "Pandas", "NumPy",
        "Excel", "Power BI", "Tableau", "Data Visualization", "Statistical Analysis",
        "Project Management", "Team Leadership", "Product Management"
    ]
    
    # Find all skills mentioned in the text
    found_skills = []
    for skill in common_skills:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            found_skills.append(skill)
    
    # Add some related skills
    if "Python" in found_skills:
        related = ["Django", "Flask", "Pandas", "NumPy"]
        found_skills.extend([s for s in related if s not in found_skills][:2])
    
    if "JavaScript" in found_skills:
        related = ["React", "Node.js", "Express", "TypeScript"]
        found_skills.extend([s for s in related if s not in found_skills][:2])
    
    # Ensure we have at least some skills
    if len(found_skills) < 5:
        found_skills = ["Python", "JavaScript", "SQL", "Git", "Agile", "Problem Solving", "Communication"]
    
    # Limit to reasonable number
    return found_skills[:15]

def extract_contact_info(text):
    """Extract contact information from resume text"""
    # Extract email
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    email = email_match.group(0) if email_match else "example@email.com"
    
    # Extract phone
    phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
    phone = phone_match.group(0) if phone_match else "555-123-4567"
    
    # Extract location
    location_match = re.search(r'\b([A-Za-z\s]+,\s*[A-Z]{2})\b', text)
    location = location_match.group(0) if location_match else "San Francisco, CA"
    
    return {
        'email': email,
        'phone': phone,
        'location': location
    }

# Job Matching Functions
def get_job_matches(parsed_resume):
    """
    Find job matches based on the parsed resume using AI analysis.
    
    Args:
        parsed_resume: Dictionary containing parsed resume information
    
    Returns:
        DataFrame: A DataFrame of matched jobs with scores
    """
    # Create a GenAI instance with the OpenAI API key
    import os
    openai_api_key = os.environ.get("OPENAI_API_KEY", "your_api_key_here")
    gen_ai = GenAI(openai_api_key)
    
    # Convert the parsed resume to a JSON string for the prompt
    resume_json = json.dumps(parsed_resume, indent=2)
    
    # In a real implementation, this would query a job database or API
    # For this demo, we'll have GPT generate relevant job listings based on the resume
    
    prompt = f"""
    Based on this parsed resume:
    
    {resume_json}
    
    Generate 10 realistic job listings that would be excellent matches for this candidate.
    
    For each job listing, include:
    1. "Title": A specific job title that matches their background
    2. "Company": A fictional company name
    3. "Location": A realistic location (mix of remote and on-site)
    4. "Salary Range": A realistic salary range based on the experience level
    5. "Match Score (%)": A percentage between 65 and 98 indicating how well the candidate matches the job
    6. "Description": A detailed job description with responsibilities and requirements
    
    The match scores should vary and be based on:
    - Skill alignment with the candidate's resume
    - Experience level match
    - Career progression potential
    
    Return the result as a JSON array of job objects.
    """
    
    try:
        # Get GPT's generated job matches
        response = gen_ai.generate_text(
            prompt=prompt,
            instructions="You are an expert job matching AI that understands both candidate qualifications and job market needs.",
            model="gpt-4o-mini",
            response_format='json_object',
            temperature=0.7
        )
        
        # Parse the JSON response
        job_listings = json.loads(response) if isinstance(response, str) else response
        
        # Convert to DataFrame
        df = pd.DataFrame(job_listings)
        
        # Sort by match score descending
        df = df.sort_values(by='Match Score (%)', ascending=False).reset_index(drop=True)
        
        return df
        
    except Exception as e:
        # Fallback to synthetic job generation if AI fails
        print(f"GPT job matching failed: {e}. Falling back to synthetic job matching.")
        
        # Extract relevant details from resume for matching
        skills = parsed_resume.get('skills', [])
        experience = parsed_resume.get('experience', [])
        
        # Extract job titles and companies from experience
        job_titles = [exp.get('title', '') for exp in experience if 'title' in exp]
        job_keywords = extract_keywords_from_titles(job_titles)
        
        # Generate job matches
        tech_jobs = generate_tech_jobs(skills, job_keywords)
        
        # Convert to DataFrame
        df = pd.DataFrame(tech_jobs)
        
        # Sort by match score descending
        df = df.sort_values(by='Match Score (%)', ascending=False).reset_index(drop=True)
        
        return df

def extract_keywords_from_titles(titles):
    """Extract relevant keywords from job titles"""
    keywords = set()
    for title in titles:
        # Extract words from title
        words = re.findall(r'\b[A-Za-z]+\b', title)
        keywords.update([word for word in words if len(word) > 2])
    
    return list(keywords)

def generate_tech_jobs(skills, keywords):
    """Generate synthetic tech job listings"""
    # Company names
    companies = [
        "TechGiant", "InnovateCorp", "DataSystems", "CloudSolutions", 
        "AI Research", "FutureTech", "DevOps Inc", "Analytics Co",
        "CyberSafe", "SmartSoft", "CodeMasters", "AppWorks",
        "Digital Dynamics", "WebFront", "MobileTech", "DataViz"
    ]
    
    # Job title templates
    title_templates = [
        "{level} {role}",
        "{role} {level}",
        "{level} {specialty} {role}",
        "{specialty} {role}",
        "{role} ({specialty})"
    ]
    
    # Components for job titles
    levels = ["Junior", "Senior", "Lead", "Principal", "Staff", ""]
    roles = ["Developer", "Engineer", "Architect", "Analyst", "Scientist", "Manager", "Specialist"]
    specialties = ["Software", "Web", "Mobile", "Frontend", "Backend", "Full Stack", 
                  "DevOps", "QA", "Data", "Cloud", "Machine Learning", "AI"]
    
    # Location options
    locations = [
        "San Francisco, CA", "New York, NY", "Austin, TX", "Seattle, WA", 
        "Boston, MA", "Chicago, IL", "Denver, CO", "Remote", 
        "Los Angeles, CA", "Atlanta, GA", "Portland, OR", "Washington, DC"
    ]
    
    # Salary ranges
    salary_ranges = [
        "$80,000 - $100,000", "$90,000 - $120,000", "$110,000 - $150,000", 
        "$130,000 - $180,000", "$150,000 - $200,000", "$180,000 - $250,000"
    ]
    
    jobs = []
    for i in range(15):  # Generate 15 job listings
        # Pick a random company
        company = random.choice(companies)
        
        # Create a job title
        level = random.choice(levels)
        role = random.choice(roles)
        specialty = random.choice(specialties)
        title_template = random.choice(title_templates)
        title = title_template.format(level=level, role=role, specialty=specialty)
        title = title.replace("  ", " ").strip()  # Fix double spaces
        
        # Pick a random location
        location = random.choice(locations)
        
        # Pick a random salary range
        salary_range = random.choice(salary_ranges)
        
        # Calculate match score based on skills and keywords
        match_score = calculate_match_score(title, company, skills, keywords)
        
        # Generate job description
        description = generate_job_description(title, company, skills)
        
        # Add job to list
        jobs.append({
            'Title': title,
            'Company': company,
            'Location': location,
            'Salary Range': salary_range,
            'Match Score (%)': match_score,
            'Description': description
        })
    
    return jobs

def calculate_match_score(title, company, skills, keywords):
    """Calculate a match score between resume and job"""
    base_score = random.randint(50, 85)  # Base random score
    
    # Adjust score based on skills match
    title_lower = title.lower()
    
    # Boost score for each skill or keyword present in the title
    boost = 0
    for skill in skills:
        if skill.lower() in title_lower:
            boost += random.randint(2, 5)
    
    for keyword in keywords:
        if keyword.lower() in title_lower:
            boost += random.randint(1, 3)
    
    # Cap the boost to keep scores realistic
    boost = min(boost, 25)
    
    # Final score capped at 98 (keeping 99-100 rare)
    score = min(base_score + boost, 98)
    
    return score

def generate_job_description(title, company, skills):
    """Generate a synthetic job description"""
    # Title components for customization
    title_lower = title.lower()
    is_senior = 'senior' in title_lower or 'lead' in title_lower or 'principal' in title_lower
    is_developer = 'developer' in title_lower or 'engineer' in title_lower
    is_data = 'data' in title_lower or 'analyst' in title_lower or 'scientist' in title_lower
    is_manager = 'manager' in title_lower or 'director' in title_lower
    
    # Create description components
    intro = f"{company} is seeking a {title} to join our growing team."
    
    if is_senior:
        experience = f"The ideal candidate has {random.randint(5, 10)}+ years of experience and can mentor junior team members."
    else:
        experience = f"We're looking for candidates with {random.randint(1, 4)}+ years of experience in the field."
    
    # Responsibilities
    responsibilities = []
    if is_developer:
        responsibilities = [
            "Design and implement new features and functionality",
            "Write clean, maintainable, and efficient code",
            "Debug and fix issues in existing applications",
            "Collaborate with cross-functional teams to define and implement solutions",
            "Participate in code reviews and knowledge sharing"
        ]
    elif is_data:
        responsibilities = [
            "Analyze large datasets to extract actionable insights",
            "Build predictive models using machine learning techniques",
            "Create data visualizations and reports for stakeholders",
            "Develop and maintain data pipelines",
            "Collaborate with business teams to identify opportunities for improvement"
        ]
    elif is_manager:
        responsibilities = [
            "Lead and manage a team of professionals",
            "Define project scope, goals, and deliverables",
            "Track project milestones and deadlines",
            "Allocate resources and manage budget",
            "Report to executives on project status and outcomes"
        ]
    else:
        responsibilities = [
            "Collaborate with cross-functional teams on project objectives",
            "Develop and implement solutions that meet business requirements",
            "Document processes and technical specifications",
            "Stay up-to-date with industry trends and technologies",
            "Participate in continuous improvement initiatives"
        ]
    
    # Select 3 random responsibilities
    selected_responsibilities = random.sample(responsibilities, 3)
    responsibilities_text = "Responsibilities:\n- " + "\n- ".join(selected_responsibilities)
    
    # Required skills
    matching_skills = []
    if is_developer:
        potential_skills = ["JavaScript", "Python", "Java", "C#", "React", "Angular", "Node.js", 
                           "AWS", "Docker", "Kubernetes", "CI/CD", "Git"]
    elif is_data:
        potential_skills = ["Python", "R", "SQL", "Pandas", "NumPy", "TensorFlow", "PyTorch", 
                           "Data Visualization", "Statistical Analysis", "Machine Learning"]
    elif is_manager:
        potential_skills = ["Project Management", "Agile", "Scrum", "Jira", "Team Leadership", 
                           "Budgeting", "Resource Planning", "Stakeholder Management"]
    else:
        potential_skills = ["Communication", "Problem Solving", "Collaboration", "Critical Thinking", 
                           "Time Management", "Attention to Detail", "Adaptability"]
    
    # Add some matching skills from the resume
    for skill in skills:
        if skill in potential_skills:
            matching_skills.append(skill)
    
    # Add some random skills from the potential list
    remaining_skills = [s for s in potential_skills if s not in matching_skills]
    random_skills = random.sample(remaining_skills, min(4, len(remaining_skills)))
    required_skills = matching_skills + random_skills
    
    # Ensure we have at least 3 skills
    while len(required_skills) < 3:
        required_skills.append(random.choice(potential_skills))
    
    # Get unique skills
    required_skills = list(set(required_skills))
    
    skills_text = "Required Skills:\n- " + "\n- ".join(required_skills[:5])
    
    # Benefits
    benefits = [
        "Competitive salary and bonus structure",
        "Health, dental, and vision insurance",
        "401(k) matching program",
        "Flexible working hours",
        "Remote work options",
        "Professional development budget",
        "Generous paid time off",
        "Modern and collaborative work environment",
        "Regular team events and activities"
    ]
    
    selected_benefits = random.sample(benefits, 3)
    benefits_text = "Benefits:\n- " + "\n- ".join(selected_benefits)
    
    # Combine all parts of the description
    description = f"{intro} {experience}\n\n{responsibilities_text}\n\n{skills_text}\n\n{benefits_text}"
    
    return description

# Resume Analysis Functions
def get_resume_tips(parsed_resume):
    """
    Generate personalized tips for improving the resume using OpenAI API.
    
    Args:
        parsed_resume: Dictionary containing parsed resume information
    
    Returns:
        list: A list of improvement suggestions
    """
    # Create a GenAI instance with your OpenAI API key
    import os
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    gen_ai = GenAI(openai_api_key)
    
    # Convert the parsed resume to a JSON string for the prompt
    resume_json = json.dumps(parsed_resume, indent=2)
    
    # Prompt for GPT to analyze the resume and provide tips
    prompt = f"""
    Analyze this parsed resume and provide specific, actionable tips for improvement:
    
    {resume_json}
    
    Provide 5-7 specific tips to improve this resume. Each tip should be:
    1. Specific and actionable
    2. Based on actual content in the resume
    3. Focused on current resume best practices
    4. Include a clear recommendation (not just pointing out a problem)
    
    Focus on areas like:
    - Professional summary effectiveness
    - Experience descriptions (achievements vs responsibilities, use of metrics)
    - Skills presentation and relevance
    - Education formatting
    - Overall structure and organization
    - ATS optimization
    
    DO NOT make up information or refer to sections that don't exist in the parsed resume.
    Return only a JSON list of tip strings without any additional text.
    """
    
    try:
        # Get GPT's analysis
        response = gen_ai.generate_text(
            prompt=prompt,
            instructions="You are an expert resume coach with years of experience in recruitment and HR.",
            model="gpt-4o-mini", 
            response_format='json_object',
            temperature=0.7
        )
        
        # Parse the JSON response
        tips = json.loads(response) if isinstance(response, str) else response
        
        # Ensure tips is a list
        if not isinstance(tips, list):
            raise ValueError("Expected a list of tips from GPT")
            
        # Return the list of tips
        return tips
        
    except Exception as e:
        print(f"Error in AI resume tips generation: {str(e)}")
        
        # Fallback to a shorter, more direct prompt
        try:
            simple_prompt = f"""
            Based on this resume data:
            {resume_json}
            
            Give 5 specific tips to improve this resume. Return as a JSON array of strings.
            """
            
            response = gen_ai.generate_text(
                prompt=simple_prompt,
                instructions="You are a resume expert. Give practical advice.",
                model="gpt-4o-mini", 
                response_format='json_object',
                temperature=0.5
            )
            
            tips = json.loads(response) if isinstance(response, str) else response
            if isinstance(tips, list):
                return tips
            else:
                raise ValueError("Response not in expected format")
                
        except Exception as inner_e:
            print(f"Fallback AI resume tips generation failed: {str(inner_e)}")
            # Return generic tips if AI analysis completely fails
            return [
                "Add more measurable achievements to your experience section using numbers and percentages.",
                "Ensure your skills section highlights the most relevant technologies for your target roles.",
                "Make your professional summary more concise and impactful by focusing on your unique value proposition.",
                "Use strong action verbs to begin bullet points in your experience section.",
                "Ensure your resume is ATS-friendly by using standard section headings and incorporating keywords from job descriptions."
            ]

def generate_improved_resume(parsed_resume, target_job=None):
    """
    Generate an improved version of the resume based on parsed content and optionally a target job.
    Uses OpenAI to enhance the content before creating the document.
    
    Args:
        parsed_resume (dict): Parsed resume information
        target_job (dict, optional): Job details to tailor the resume toward
    
    Returns:
        bytes: The binary DOCX file content
    """
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    from io import BytesIO
    import os
    import json
    
    # Create a GenAI instance with the OpenAI API key
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    gen_ai = GenAI(openai_api_key)
    
    # If we have a target job, get enhancements tailored to that job
    if target_job:
        job_title = target_job.get('Title', '')
        job_company = target_job.get('Company', '')
        job_description = target_job.get('Description', '')
        
        prompt = f"""
        I need to improve this resume to target a {job_title} position at {job_company}.
        
        Here's the job description:
        {job_description}
        
        Here's the current resume data:
        {json.dumps(parsed_resume, indent=2)}
        
        Please provide the following improvements in JSON format:
        1. An improved professional summary (1-3 sentences) that highlights relevant experience for this specific job
        2. Enhanced bullet points for each experience entry that emphasize relevant achievements
        3. A re-ordered and enhanced skills list that prioritizes skills mentioned in the job description
        
        Return only the JSON object with these three improvements.
        """
        
        try:
            # Get GPT's enhancements
            response = gen_ai.generate_text(
                prompt=prompt,
                instructions="You are an expert resume writer specializing in tailoring resumes to specific job descriptions. Focus on highlighting relevant experience and skills that match the target job.",
                model="gpt-4o-mini",
                response_format='json_object',
                temperature=0.5
            )
            
            # Parse the JSON response
            improvements = json.loads(response) if isinstance(response, str) else response
            
            # Apply the improvements to the parsed resume
            if 'summary' in improvements:
                parsed_resume['summary'] = improvements['summary']
            
            if 'experience' in improvements and isinstance(improvements['experience'], list):
                # Match and update experience bullet points
                for i, exp in enumerate(parsed_resume['experience']):
                    if i < len(improvements['experience']):
                        if 'bullet_points' in improvements['experience'][i]:
                            exp['description'] = '\n'.join(improvements['experience'][i]['bullet_points'])
            
            if 'skills' in improvements and isinstance(improvements['skills'], list):
                parsed_resume['skills'] = improvements['skills']
                
        except Exception as e:
            print(f"Error in AI resume enhancements: {str(e)}")
            # Continue with original resume if enhancement fails
    
    # Create a new Document
    doc = Document()
    
    # Set margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(0.5)
        section.bottom_margin = Inches(0.5)
        section.left_margin = Inches(0.5)
        section.right_margin = Inches(0.5)
    
    # Contact Information
    contact = parsed_resume.get('contact', {})
    name = contact.get('name', 'Your Name')
    
    # Name as title
    name_heading = doc.add_heading('', level=0)
    name_run = name_heading.add_run(name)
    name_run.font.size = Pt(18)
    name_run.font.color.rgb = RGBColor(0, 0, 0)
    name_heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Contact info
    contact_info = doc.add_paragraph()
    contact_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    contact_details = []
    
    if 'email' in contact and contact['email']:
        contact_details.append(contact['email'])
    if 'phone' in contact and contact['phone']:
        contact_details.append(contact['phone'])
    if 'location' in contact and contact['location']:
        contact_details.append(contact['location'])
    if 'linkedin' in contact and contact['linkedin']:
        contact_details.append(contact['linkedin'])
    
    contact_info.add_run(' | '.join(contact_details))
    
    # Add a line after contact info
    doc.add_paragraph().add_run('_' * 100)
    
    # Professional Summary
    if parsed_resume.get('summary'):
        heading = doc.add_heading('Professional Summary', level=1)
        heading.style.font.color.rgb = RGBColor(33, 64, 95)
        doc.add_paragraph(parsed_resume['summary'])
    
    # Work Experience
    experience = parsed_resume.get('experience', [])
    if experience:
        heading = doc.add_heading('Experience', level=1)
        heading.style.font.color.rgb = RGBColor(33, 64, 95)
        
        for exp in experience:
            # Job title and company
            p = doc.add_paragraph()
            company_line = p.add_run(f"{exp.get('title', 'Role')} | {exp.get('company', 'Company')}")
            company_line.bold = True
            
            # Dates
            date_text = f"{exp.get('start_date', '')} - {exp.get('end_date', 'Present')}"
            p.add_run(f" | {date_text}").italic = True
            
            # Description/bullet points
            if 'description' in exp and exp['description']:
                description = exp['description']
                # Split description into bullet points if it contains line breaks
                bullet_points = description.split('\n')
                for point in bullet_points:
                    if point.strip():  # Skip empty lines
                        bullet = doc.add_paragraph(point.strip(), style='List Bullet')
                        # Adjust bullet paragraph formatting
                        bullet.paragraph_format.left_indent = Inches(0.25)
    
    # Skills
    skills = parsed_resume.get('skills', [])
    if skills:
        heading = doc.add_heading('Skills', level=1)
        heading.style.font.color.rgb = RGBColor(33, 64, 95)
        
        # Create a skill category dictionary if possible
        skill_categories = {}
        
        # Try to categorize skills
        tech_skills = [s for s in skills if s.lower() in [
            'python', 'java', 'javascript', 'sql', 'c++', 'typescript', 
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring',
            'docker', 'kubernetes', 'aws', 'azure', 'gcp', 'git', 'tensorflow',
            'pytorch', 'pandas', 'numpy', 'swift', 'go', 'rust'
        ]]
        
        soft_skills = [s for s in skills if s.lower() in [
            'communication', 'leadership', 'teamwork', 'problem solving',
            'critical thinking', 'time management', 'adaptability', 'creativity',
            'project management', 'collaboration', 'presentation', 'negotiation'
        ]]
        
        tools = [s for s in skills if s.lower() in [
            'jira', 'confluence', 'trello', 'slack', 'github', 'gitlab', 'bitbucket',
            'excel', 'powerpoint', 'word', 'adobe', 'photoshop', 'illustrator',
            'tableau', 'power bi', 'looker', 'figma', 'sketch'
        ]]
        
        # Assign categorized skills
        if tech_skills:
            skill_categories['Technical Skills'] = tech_skills
        if soft_skills:
            skill_categories['Soft Skills'] = soft_skills
        if tools:
            skill_categories['Tools & Platforms'] = tools
            
        # Add remaining skills to "Other Skills"
        other_skills = [s for s in skills if s not in tech_skills and s not in soft_skills and s not in tools]
        if other_skills:
            skill_categories['Other Skills'] = other_skills
            
        # If we were able to categorize, display by category
        if skill_categories:
            for category, category_skills in skill_categories.items():
                p = doc.add_paragraph()
                p.add_run(f"{category}: ").bold = True
                p.add_run(", ".join(category_skills))
        else:
            # Just list all skills
            doc.add_paragraph(", ".join(skills))
    
    # Education
    education = parsed_resume.get('education', [])
    if education:
        heading = doc.add_heading('Education', level=1)
        heading.style.font.color.rgb = RGBColor(33, 64, 95)
        
        for edu in education:
            p = doc.add_paragraph()
            
            # Degree and institution
            edu_line = p.add_run(f"{edu.get('degree', 'Degree')} | {edu.get('institution', 'Institution')}")
            edu_line.bold = True
            
            # Dates
            date_text = f"{edu.get('start_date', '')} - {edu.get('end_date', '')}"
            p.add_run(f" | {date_text}").italic = True
            
            # GPA if available
            if 'gpa' in edu and edu['gpa']:
                p.add_run(f" | GPA: {edu['gpa']}")
                
            # Additional details if available
            if 'details' in edu and edu['details']:
                details = doc.add_paragraph(edu['details'])
                details.paragraph_format.left_indent = Inches(0.25)
    
    # Certifications (if available)
    if 'certifications' in parsed_resume and parsed_resume['certifications']:
        heading = doc.add_heading('Certifications', level=1)
        heading.style.font.color.rgb = RGBColor(33, 64, 95)
        
        for cert in parsed_resume['certifications']:
            if isinstance(cert, dict):
                p = doc.add_paragraph()
                p.add_run(f"{cert.get('name', '')}").bold = True
                if 'issuer' in cert:
                    p.add_run(f" | {cert['issuer']}")
                if 'date' in cert:
                    p.add_run(f" | {cert['date']}").italic = True
            elif isinstance(cert, str):
                doc.add_paragraph(cert)
    
    # Projects (if available)
    if 'projects' in parsed_resume and parsed_resume['projects']:
        heading = doc.add_heading('Projects', level=1)
        heading.style.font.color.rgb = RGBColor(33, 64, 95)
        
        for project in parsed_resume['projects']:
            if isinstance(project, dict):
                p = doc.add_paragraph()
                p.add_run(f"{project.get('name', 'Project')}").bold = True
                if 'date' in project:
                    p.add_run(f" | {project['date']}").italic = True
                
                if 'description' in project:
                    desc = doc.add_paragraph(project['description'])
                    desc.paragraph_format.left_indent = Inches(0.25)
            elif isinstance(project, str):
                doc.add_paragraph(project)
    
    # If resume was tailored for a specific job, add a note
    if target_job:
        doc.add_page_break()
        doc.add_heading('Resume Tailored For', level=1)
        p = doc.add_paragraph()
        p.add_run(f"{target_job.get('Title')} at {target_job.get('Company')}").bold = True
        
        if 'Description' in target_job:
            doc.add_paragraph('Job Description:')
            doc.add_paragraph(target_job.get('Description'))
            
        # Add tailoring notes
        doc.add_heading('Tailoring Notes', level=2)
        doc.add_paragraph('This resume has been optimized to highlight your relevant skills and experience for this specific position. Key adjustments include:')
        
        bullet_points = [
            'Emphasized experience most relevant to the target role',
            'Highlighted skills mentioned in the job description',
            'Tailored professional summary to match job requirements',
            'Used industry-specific keywords to increase ATS match score'
        ]
        
        for point in bullet_points:
            doc.add_paragraph(point, style='List Bullet')
    
    # Export to BytesIO
    resume_bytes = BytesIO()
    doc.save(resume_bytes)
    resume_bytes.seek(0)
    return resume_bytes.read()

# AI Career Agent Functions
def get_career_agent_prompt(parsed_resume):
    """
    Generate a personalized prompt for the AI career advisor based on the parsed resume.
    
    Args:
        parsed_resume: Dictionary containing parsed resume information
    
    Returns:
        str: A prompt string defining the AI's behavior and knowledge
    """
    # Extract key information from the parsed resume
    contact = parsed_resume.get('contact', {})
    name = contact.get('name', 'the user')
    
    # Get skills
    skills = parsed_resume.get('skills', [])
    skills_text = ', '.join(skills[:5]) if skills else 'not specified'
    
    # Get experience information
    experience = parsed_resume.get('experience', [])
    years_of_experience = len(experience)
    
    # Determine experience level
    if years_of_experience >= 7:
        experience_level = "senior professional"
    elif years_of_experience >= 3:
        experience_level = "mid-level professional"
    else:
        experience_level = "early-career professional"
    
    # Get most recent job title
    recent_title = experience[0].get('title', 'professional') if experience else "professional"
    
    # Get education
    education = parsed_resume.get('education', [])
    education_text = ''
    if education:
        degree = education[0].get('degree', '')
        institution = education[0].get('institution', '')
        if degree and institution:
            education_text = f" with {degree} from {institution}"
    
    # Create the detailed persona prompt
    prompt = f"""
    You are an expert career coach and job search advisor named NextSteps Career Agent. You're talking to {name}, a {experience_level} who most recently worked as a {recent_title}{education_text}. 
    Their key skills include {skills_text}.
    
    Based on their resume, here's more context about their background:
    - Experience: {years_of_experience} positions in their work history
    - Most recent role: {recent_title}
    - Education background: {education_text if education_text else 'Not specified in detail'}
    
    Your role is to provide personalized career advice, resume tips, job search strategies, and interview preparation guidance. You should be:
    
    1. Supportive and encouraging while providing honest, constructive feedback
    2. Specific in your advice, referencing their actual experience, skills, and background
    3. Knowledgeable about industry trends and job market realities for their field
    4. Practical in your suggestions - focus on actionable steps they can take immediately
    5. Empathetic to the challenges of job searching
    
    Provide concrete tips and strategies tailored to their specific situation. For example:
    
    - If they ask about improving their resume, reference specific sections from their experience
    - If they ask about job searching, consider their specific skills and experience level
    - If they ask about interviews, provide advice relevant to their industry and background
    
    Keep your responses concise, practical, and positive. Remember that you are their career partner, helping them navigate their professional journey with expertise and care.
    """
    
    return prompt

def get_ai_response(persona_prompt, messages):
    """
    Generate an AI response for the career agent chat using OpenAI API.
    
    Args:
        persona_prompt: String defining the AI's behavior
        messages: List of message dictionaries in OpenAI format
    
    Returns:
        str: The AI's response
    """
    # Create a GenAI instance with your OpenAI API key
    import os
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    gen_ai = GenAI(openai_api_key)
    
    try:
        # Create a formatted chat history for GPT
        chat_history = []
        
        # Add previous messages (excluding the most recent user message)
        for msg in messages[:-1]:
            chat_history.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Get the most recent user message
        user_message = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else ""
        
        # Generate response using the chat completion API
        response = gen_ai.generate_chat_response(
            chat_history=chat_history,
            user_message=user_message,
            instructions=persona_prompt,
            model="gpt-4o-mini",
            output_type='text'
        )
        
        return response
        
    except Exception as e:
        print(f"Error in AI career agent response: {str(e)}")
        
        # Simplified fallback if full chat history fails
        try:
            # Just use the most recent message
            user_message = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else ""
            
            if not user_message:
                return "I'm here to help with your career questions. What would you like to discuss about your job search or career development?"
            
            prompt = f"""
            {persona_prompt}
            
            User question: {user_message}
            
            Provide a helpful, concise response.
            """
            
            response = gen_ai.generate_text(
                prompt=prompt,
                instructions="You are a helpful career advisor. Keep your response concise and practical.",
                model="gpt-4o-mini",
                output_type='text',
                temperature=0.7
            )
            
            return response
            
        except Exception as inner_e:
            print(f"Simplified AI career agent response failed: {str(inner_e)}")
            
            # Hardcoded fallback responses for various message types
            if any(word in user_message.lower() for word in ['hi', 'hello', 'hey', 'greet']):
                return "Hello! I'm your NextSteps Career Agent. How can I help with your job search or career development today?"
            
            if any(word in user_message.lower() for word in ['resume', 'cv']):
                return "Looking at your resume, I recommend highlighting your achievements with specific metrics and tailoring your skills section to match the jobs you're applying for. Would you like more specific resume advice?"
            
            if any(word in user_message.lower() for word in ['interview', 'interviewing']):
                return "For interviews, prepare by researching the company thoroughly and practicing the STAR method (Situation, Task, Action, Result) for behavioral questions. Would you like me to explain how to use this method effectively?"
            
            if any(word in user_message.lower() for word in ['thank', 'thanks']):
                return "You're welcome! I'm here to support your career journey. Let me know if you have any other questions."
            
            # Default response
            return "I'm here to help with your career development. I can provide advice on resume optimization, job search strategies, interview preparation, and career planning. What specific aspect would you like to discuss?"    

from docx import Document
from io import BytesIO

def check_openai_key():
    """Check if the OpenAI API key is valid."""
    try:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return False
        openai.api_key = key
        # Minimal test to check key validity
        openai.Model.list()
        return True
    except Exception:
        return False

