Chat Analyzer is a web application that analyzes chat conversations from Telegram, WhatsApp, or plain text using local language models via Ollama. 

It calculates message statistics, detects sentiment, identifies conflicts, and generates a structured summary of the conversation. 

It supports Telegram JSON exports, WhatsApp text exports, and plain text in the format "Author: message", 

and all processing is done locally without external APIs. To run the project, clone the repository using 

`git clone https://github.com/vainleet/LocalAiChatAnalyser` 

and go into the folder with 

`cd LocalAiChatAnalyser`

Install dependencies with `pip install flask ollama`. 

Install Ollama: on Windows download the installer from https://ollama.com/download and run it. 

After installation, start Ollama with 

`ollama serve`

Download the default model with 

`ollama pull gemma3:4b` 

Start the application using `main.py` and open your browser at `http://localhost:5000` 

Upload a chat file, optionally select a model, and run the analysis. 

The API supports `GET /health` to check Ollama status and available models, and `POST /analyze` to upload a chat file and receive JSON output.

To download chat from Telegram, go to chat and click three dots, Export chat history, disable everything and change output file to JSON!!!!
