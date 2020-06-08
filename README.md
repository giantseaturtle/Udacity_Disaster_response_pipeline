# Disaster Response Pipeline Project

### Motivation
This project is following the requirement of Data Scientist Nanodegree of Udacity to apply ETL pipeline to analyze disaster data from Figure Eight, and build a ML model for an API that can classify disaster messages.

### Required libraries
This project was written by Python 3.7, including following packages: 
- Data processing and machine learning: Numpy, Pandas, Sciki-Learn, Plotly
- Natural language processing: NLTK
- SQLite Database: SQLalchemy
- Web App Deployment: Flask

### Description of files
- Screenshot: Screenshot the layout of web app
- app: Deploy and visualize the cleaned database, as well as the trained ML model
- Data: Load the disaster_messages and disaster_categories datasets, clean and merge them into SQLite database
- Model: Import the cleaned data, transform it with NLP, run a machine learning model on it with GridSearchCV.

### Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/, you will get following web app.
![screenshot1](https://user-images.githubusercontent.com/49320590/83955013-08487500-a814-11ea-8f4b-5ebb6abe224e.png)

### Example
In the web app, it will classify the import text message in categories, the results could be sent to any related organizations, so the service or help can be delivered in time. 

For instance, typing "We have quarantined in our apartment for seven days, we need food!", the message will be tagged as "Related", "Request", "Food", and "Direct Report", this message can be accurately assigned to food-aid related organization.
![screenshot2](https://user-images.githubusercontent.com/49320590/83955014-10a0b000-a814-11ea-91a2-236b62938304.png)

### License
[MIT License](https://opensource.org/licenses/MIT)

### Credit
[Udacity](https://www.udacity.com/) for providing the web app template

[Figure Eight](https://appen.com/) for providing the dataset
