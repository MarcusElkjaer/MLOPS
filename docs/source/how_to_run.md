# How to Run Reddit Forecast

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/reddit-forecast/reddit-forecast.git
   cd reddit-forecast
   ```
2. **Set Up a Virtual Environment** (Optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```
4. **Pull Data with DVC**:

   ```bash
   dvc pull
   ```
## Running the Project

### 1. **Preprocess Data**

   Run the preprocessing script to clean and prepare the data:

   ```bash
   python src/data.py
   ```
    The preprocessed data will be saved in the `data/processed/` folder.
### 2. **Run the Model**

   Run the machine learning model using the processed data:

   ```bash
   python src/model.py
   ```
   This will generate model files in the `models/` directory.

### 3. **Train the Model**

   Train the machine learning model using the processed data:

   ```bash
   python src/train.py
   ```
   This will generate model files in the `models/` directory.
### 4. **Run the API**

   Launch the API to interact with Reddit Forecast programmatically:

   ```bash
   uvicorn reddit_forecast.api:app --reload --port 8000
   ```

   The API will be available at `http://127.0.0.1:8000`.

### 5. **Test the Project**

   Run unit tests to ensure the project is functioning correctly:

   ```bash
   pytest tests/
   ```
