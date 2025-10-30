# Vietnamese NER for COVID-19 Medical Entities using PhoBERT

> **Repository**: [https://github.com/doananhhung/NER\_Covid19](https://www.google.com/search?q=https://github.com/doananhhung/NER_Covid19)

This project focuses on Named Entity Recognition (NER) for extracting specific medical and epidemiological entities from Vietnamese texts related to the COVID-19 pandemic. The model is built by fine-tuning **PhoBERT**, a state-of-the-art monolingual BERT-based model for Vietnamese.

The project includes scripts for data exploration, training, evaluation, and a simple web-based demo application using Streamlit.

## ✨ Features

  * **High-Performance NER**: Fine-tunes PhoBERT (`vinai/phobert-base`) for a specialized Vietnamese NER task.
  * **Comprehensive Entity Recognition**: Identifies 10 types of entities relevant to medical and COVID-19 contexts.
  * **Structured Project**: Well-organized codebase with clear separation of concerns for configuration, data handling, training, and inference.
  * **Reproducibility**: Includes a requirements file and a fixed random seed for consistent results.
  * **Interactive Demo**: Comes with a Streamlit web application for easy testing and visualization of results.
  * **Colab Ready**: Provides a Jupyter notebook for training the model on Google Colab's free GPU resources.

## 🏷️ Entities Recognized

The model is trained to recognize and classify the following entities:

| Tag                       | Description                               |
| ------------------------- | ----------------------------------------- |
| `PATIENT_ID`              | Patient identification code               |
| `SYMPTOM_AND_DISEASE`     | Symptoms and diseases mentioned           |
| `LOCATION`                | Geographical locations (cities, hospitals)|
| `DATE`                    | Dates of events (e.g., admission date)    |
| `ORGANIZATION`            | Organizations involved (e.g., Ministry of Health) |
| `AGE`                     | Age of the patient                        |
| `GENDER`                  | Gender of the patient                     |
| `NAME`                    | Names of individuals                      |
| `TRANSPORTATION`          | Mode of transportation used               |
| `JOB`                     | Occupation of the patient                 |

*This list is defined in `src/config.py`.*

## 🚀 Getting Started

### Prerequisites

  * Python 3.8+
  * PyTorch
  * Git

### 1\. Clone the Repository

```bash
git clone https://github.com/doananhhung/NER_Covid19.git
cd NER_Covid19
```

### 2\. Set Up a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 3\. Install Dependencies

Install all the required packages using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4\. Download Pre-trained Model (Optional)

If you want to run the demo app without training the model yourself, you can download the pre-trained model from the link below:

  * **[Download Model from Google Drive](https://drive.google.com/drive/folders/1GNf_xUUrswxe3feUWCaTyyLbzFnLfLHS?usp=drive_link)**

After downloading, unzip the file and place the resulting `phobert-ner-covid` folder inside the `models/` directory.

### 5\. Download Data

This project uses the `PhoNER_COVID19` dataset. Please place the `train_word.json`, `dev_word.json`, and `test_word.json` files inside the `data/raw/PhoNER_COVID19/` directory.

## 🛠️ Usage

The project is structured with a clear workflow from training to inference.

### 1\. Data Exploration (Optional)

To understand the dataset statistics (label distribution, sentence length), you can explore the `notebooks/Data_Exploration.ipynb` notebook. This can help in fine-tuning hyperparameters.

### 2\. Training the Model

To train the NER model, run the `train.py` script from the project's root directory. The script will handle data loading, model initialization, training, and validation. The best-performing model (based on F1-score on the dev set) will be saved to `models/phobert-ner-covid`.

```bash
python src/train.py
```

All hyperparameters like batch size, learning rate, and epochs can be modified in `src/config.py`.

### 3\. Evaluating the Model

After training, you can evaluate the final model's performance on the test set by running:

```bash
python src/evaluate.py
```

This will print a detailed classification report with precision, recall, and F1-score for each entity type.

### 4\. Running the Interactive Demo (Streamlit App)

To visualize the model's predictions in an interactive web interface, run the Streamlit app:

```bash
streamlit run app/app.py
```

This will launch a local web server. You can input a sentence, and the app will highlight the recognized entities with their corresponding tags.

### 5\. Standalone Inference

For programmatic predictions, you can use the `NERPredictor` class from `src/inference.py`. The script also serves as a command-line demo.

```bash
python src/inference.py
```

## 📂 Project Structure

```
NER_Covid19/
├── app/                  # Source code for the Streamlit web application
│   ├── app.py            # Main Streamlit app script
│   └── utils.py          # Utility functions for the app (e.g., entity rendering)
├── data/                 # Dataset files
│   └── raw/PhoNER_COVID19/ # Raw data files (train, dev, test .json)
├── models/               # Saved model checkpoints
│   └── phobert-ner-covid/  # Saved fine-tuned model and tokenizer
├── notebooks/            # Jupyter notebooks for exploration and training
│   ├── Data_Exploration.ipynb
│   └── Train_on_Colab_basic.ipynb
├── src/                  # Core source code
│   ├── config.py         # Central configuration and hyperparameters
│   ├── dataset.py        # PyTorch Dataset class for NER
│   ├── evaluate.py       # Script for evaluating the model on the test set
│   ├── inference.py      # Script and class for making predictions
│   └── train.py          # Main training script
├── .gitignore            # Files to be ignored by Git
├── README.md             # This file
└── requirements.txt      # Python dependencies
```

## 💻 Tech Stack

  * **Core Libraries**: PyTorch, Transformers, Torch
  * **Data Handling**: Pandas
  * **Evaluation**: seqeval
  * **Web App**: Streamlit
