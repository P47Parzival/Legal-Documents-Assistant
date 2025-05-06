# ⚖️ Legal Documents Assistant

A powerful AI-powered assistant designed to help legal professionals and individuals navigate through legal documents efficiently. This tool leverages advanced natural language processing and document analysis capabilities to provide quick insights and answers from legal documents.


## Features

- **Document Analysis**: Upload and analyze legal documents with ease
- **Smart Search**: Find relevant information quickly using natural language queries
- **Interactive Interface**: User-friendly interface for seamless document interaction
- **AI-Powered Insights**: Get intelligent responses to your legal document queries

# Images

![Legal Assistant Interface](https://github.com/P47Parzival/Legal-Documents-Assistant/blob/main/Images/img1.png?raw=true)

![Interactive Interface](https://github.com/P47Parzival/Legal-Documents-Assistant/blob/main/Images/img3.png?raw=true)

![Document Analysis](https://github.com/P47Parzival/Legal-Documents-Assistant/blob/main/Images/img2.png?raw=true)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Legal-Documents-Assistant.git
cd Legal-Documents-Assistant
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
Create a `.env` file in the root directory and add your necessary API keys:
```
GOOGLE_API_KEY=your_google_api_key
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload your legal documents and start interacting with the assistant



## Technologies Used

- Streamlit
- LangChain
- Google Generative AI
- FAISS for vector storage
- PyMuPDF for PDF processing

## Requirements

The project requires Python 3.8+ and the following main dependencies:
- langchain==0.3.25
- langchain-community==0.3.23
- langchain-google-genai==2.1.4
- streamlit==1.45.0
- PyMuPDF==1.25.5
- faiss-cpu==1.11.0
- google-generativeai==0.8.5

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape this project
- Special thanks to the open-source community for the amazing tools and libraries 