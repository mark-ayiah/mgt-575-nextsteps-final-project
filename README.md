# 🚀 NextSteps - AI Career Assistant

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

NextSteps is an AI-powered career assistant built with Streamlit that helps users optimize their job search by analyzing resumes, matching them to real-world job opportunities, improving resume quality, and offering personalized career coaching through an AI agent.

## ✨ Features

- **Resume Parsing**: Upload your resume (PDF or DOCX) and get a detailed AI-powered analysis of your skills, experience, and education.
- **Census-Based Job Matching**: Get matched to official job roles from the SOC 2018 Census dataset based on your resume content and match score.
- **Real Job Opportunities**: Receive realistic AI-generated job postings (with company names, salaries, and locations) customized for your profile.
- **Resume Feedback**: Get 5 actionable tips to immediately strengthen your resume.
- **Enhanced Resume Generation**: Download an AI-enhanced, ATS-optimized version of your resume (in Word format).
- **Career Agent Chat**: Chat with an AI career coach that provides live, tailored advice based on your profile and career goals.
- **Dynamic Resume Upload and Reset**: Seamlessly upload new resumes and restart your analysis without breaking the app.
- **Custom Career Suggestions**: Ask the AI agent about specific job titles or industries and get practical preparation tips.

## 📦 Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/nextsteps.git
   cd nextsteps
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - **Option 1**: Set it as an environment variable:
     ```bash
     export OPENAI_API_KEY=your_api_key_here
     ```
   - **Option 2**: Create a `.streamlit/secrets.toml` file with the following content:
     ```toml
     [openai]
     api_key = "your_api_key_here"
     ```
   - **Option 3**: Enter your API key manually inside the app when prompted.

## 🚀 Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open your browser at the provided local URL (typically http://localhost:8501).

3. Upload your resume and explore:
   - View matches to official Census job roles
   - Discover realistic job listings
   - Improve your resume
   - Chat with your career agent for coaching

## 📁 Project Structure

- `app.py`: Main Streamlit app with page navigation and feature logic.
- `utils.py`: Utility functions for text extraction, AI interactions, and resume analysis.
- `config.py`: Handles API key loading from environment variables or secrets.
- `requirements.txt`: Python package dependencies.

## 🛠 Future Enhancements

- Direct linking to real job board listings (Indeed, LinkedIn, etc.)
- AI-crafted interview question simulations
- Expandable career agent personalities (Mentor, Optimist, Realist)
- Career trajectory visualizations and salary growth projections
- Automated cover letter generation tailored to jobs

## 🧰 Technologies Used

- **Streamlit** — Web app framework
- **OpenAI API** — GPT-based resume parsing, job matching, and career advising
- **pdfplumber** and **docx2txt** — Resume file text extraction
- **Pandas** — Data manipulation for SOC job matching
- **Regex** — Intelligent text cleaning and formatting

## 📜 License

This project is licensed under the MIT License — see the LICENSE file for details.

## 🙌 Acknowledgments

- Built as a next-generation AI career assistant prototype.
- Special thanks to OpenAI for the API technologies that made this possible!

## 👥 Contributors

- Mark Ayiah
- Isabel Nuño
