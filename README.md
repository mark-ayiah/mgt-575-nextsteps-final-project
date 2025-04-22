# NextSteps - AI Career Assistant

NextSteps is an AI-powered career assistant built with Streamlit that helps users optimize their job search by analyzing resumes, matching with relevant jobs, providing personalized resume feedback, and offering career coaching through an AI agent.

## Features

- **Resume Parsing**: Upload your resume (PDF or DOCX) and get an AI-powered analysis of your experience, skills, and qualifications
- **Job Matching**: Get matched with relevant job listings based on your resume content
- **Resume Feedback**: Receive personalized tips to improve your resume for better job prospects
- **Career Agent**: Chat with an AI career coach that provides personalized advice based on your resume
- **Enhanced Resume Generation**: Create an improved version of your resume with AI optimization

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/nextsteps.git
   cd nextsteps
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Option 1: Set it as an environment variable: `export OPENAI_API_KEY=your_api_key_here`
   - Option 2: Create a Streamlit secrets.toml file in `.streamlit/secrets.toml` with:
     ```
     [openai]
     api_key = "your_api_key_here"
     ```
   - Option 3: Enter it directly in the app when prompted

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser to the URL displayed in the terminal (usually http://localhost:8501)

3. Upload your resume and explore the features

## Project Structure

- `app.py`: Main Streamlit application
- `utils.py`: Helper functions for resume parsing, job matching, and AI interactions
- `config.py`: Configuration handling (API key management)
- `requirements.txt`: Required Python packages

## Future Enhancements

- Job listing URL parsing and custom matching
- AI-generated resume rewrites tailored to specific positions
- Different AI coaching personas (Mentor, Tough Love Coach, etc.)
- Direct job application functionality through API integrations
- Interview simulation and preparation
- Career trajectory planning and visualization

## Technologies Used

- Streamlit for the web interface
- OpenAI API for natural language processing and AI capabilities
- PyPDF2 and python-docx for document processing
- Pandas for data handling

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This project was developed as a prototype for an AI-powered career assistant
- Thanks to OpenAI for providing the API that powers the AI functionality

## Contributors

- Mark Ayiah
- Isabel Nuno
