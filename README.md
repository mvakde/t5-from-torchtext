(Will update this soon)  

### Initialisation
- Clone this repository
- Use python version 3.12.7 (If you have pyenv installed, run `pyenv local 3.12.7`. If it throws an error saying not installed, run `pyenv install 3.12.7` first
- Create virtual environment with necessary packages:
    - `python3 -m venv venv`
    - `source venv/bin/activate`
    - `pip install -r requirements.txt`

### Running:
- `python compare.py` to check whether the implementation of t5 matches huggingface's model exactly
- `python inference.py` for inference
    - Modify this file as per your whims to test how t5 works
