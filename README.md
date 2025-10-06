## Usage

1. Clone this repo :
```bash
git clone https://github.com/sawsansalameh222/Twitter_Sentiment_classification.git
```
```bash
cd Twitter_Sentiment_classification
```
2. Create and activate the virtual environment (Windows):
```bash
python -m venv venv
venv\Scripts\activate
```


On Mac/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

3.  Make sure `model.pkl` is in the repo folder.
4.  Run the predictor:
  ```bash
  python predict.py ```

5. If `vectorizer.pkl` is missing, it will be downloaded automatically from Google Drive.

## Model
The trained model file is `model.pkl`, included in this repository.

