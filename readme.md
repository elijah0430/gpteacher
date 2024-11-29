```markdown
# GPTeacher Assistant

GPTeacher Assistant is a Streamlit-based web application designed to help users solve programming tests by providing step-by-step guidance and interactive assistance using OpenAI's GPT-4 model. The application allows users to select a programming test, view its details, and engage in a conversational interface to receive tailored advice and solutions.

## Features

- **Test Selection:** Choose from a variety of programming tests loaded from a JSON file.
- **Interactive Chat:** Engage in a conversation with an AI assistant that provides guidance and answers to your coding questions.
- **Step-by-Step Plans:** Receive long-term dialogue plans to systematically approach and solve programming problems.
- **Conversation History:** Maintains a history of your interactions for context-aware assistance.
- **Multilingual Support:** The assistant responds in Korean to cater to Korean-speaking users.

## Screenshots

## Installation

### Prerequisites

- **Python 3.7 or higher** installed on your system.
- **OpenAI API Key:** You need an OpenAI API key to use the GPT-4 model. You can obtain one from [OpenAI](https://platform.openai.com/).

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/gpteacher-assistant.git
   cd gpteacher-assistant
   ```

2. **Create a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Create a `.env` file in the root directory and add your OpenAI API key:

   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   Replace `your_openai_api_key_here` with your actual OpenAI API key.

5. **Prepare Data**

   Ensure that the `data.json` file containing your programming tests is present in the root directory. The structure should be as follows:

   ```json
   [
       {
           "title": "Sample Test 1",
           "text": "Description of Sample Test 1."
       },
       {
           "title": "Sample Test 2",
           "text": "Description of Sample Test 2."
       }
       // Add more tests as needed
   ]
   ```

6. **Create Results Directory**

   Create a `results` directory to store conversation histories:

   ```bash
   mkdir results
   ```

## Usage

Run the Streamlit application using the following command:

```bash
streamlit run streamlit_app.py
```

This will start the application and open it in your default web browser. If it doesn't open automatically, navigate to `http://localhost:8501` in your browser.

## Project Structure

```
gpteacher-assistant/
│
├── data.json                  # JSON file containing programming tests
├── results/                   # Directory to store conversation histories
│
├── streamlit_app.py           # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (not committed)
├── README.md                  # Project documentation
└── screenshots/               # (Optional) Screenshots for README
```

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add some feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or suggestions, please open an issue or contact [your.email@example.com](mailto:your.email@example.com).