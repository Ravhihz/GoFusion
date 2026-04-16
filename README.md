# GoFusion — Fusion Sentiment Analysis

GoFusion is a web-based sentiment analysis platform built with Django, designed to process and classify Indonesian tweets using a Support Vector Machine (SVM) algorithm implemented from scratch. It provides an end-to-end workflow accessible through a clean web interface.

## Features

- **Dataset Management** — Upload and manage tweet datasets in CSV format
- **Preprocessing** — Tokenization, normalization, and stopword removal for Indonesian text
- **Preliminary Check** — Filter relevant and irrelevant tweets manually or by keyword
- **Sampling** — Sample tweets from the dataset for training
- **Labeling** — Manually label tweets with sentiment (positive, negative, neutral)
- **Model Training** — Train SVM model with custom parameters
- **Prediction** — Predict sentiment on new data using trained models
- **Analytics** — Visualize sentiment distribution and model performance

## Tech Stack

- **Backend** — Python, Django 4.2
- **Machine Learning** — SVM from scratch, Gensim, NLTK, Sastrawi
- **Database** — MySQL (via PyMySQL)
- **Frontend** — HTML, CSS, JavaScript (Django Templates)
- **Data Processing** — Pandas, NumPy, Matplotlib, Seaborn

## Installation

1. Clone the repository
```bash
   git clone https://github.com/Ravhihz/GoFusion.git
   cd GoFusion
```

2. Install dependencies
```bash
   pip install -r requirements.txt
```

3. Configure environment — create a `.env` file in the root directory
- SECRET_KEY=your-secret-key
- DB_NAME=your-db-name
- DB_USER=your-db-user
- DB_PASSWORD=your-db-password
- DB_HOST=localhost
- DB_PORT=3306

4. Run migrations
```bash
   python manage.py migrate
```

5. Start the development server
```bash
   python manage.py runserver
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
