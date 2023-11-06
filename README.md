# Dependencies
1. python3
1. openAI chatGPT API key:
Store this in a .env file in the root of the project in the following format:
  OPENAI_API_KEY=skXXXXX...
1. virtualenv: `python3 -m pip install --user virtualenv`

# Usage

```bash
virtualenv .venv
pip install --upgrade pip
source ./venv/bin/activate
pip3 install -r requirements.txt
python3 multi.py
```
