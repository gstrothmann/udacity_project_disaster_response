# Disaster Response Pipeline Project

### About the Project

The goal of this project is an interactive webapp that is able to classify messages sent from areas in which natural disasters occured.
The classification algorithm used is pretrained with a labeled dataset that consist of more than 20.000 messages and 36 possible labels, such as "request", "medical_aid" or "weather_related". 

### Structure of the project

The folder structure of the project looks as follows:
- app
- - templates
- - - go.html
- - - master.html
- - run.py
- data
- - disaster_categories.csv
- - disaster_messages.csv
- - process_data.py
- models
- - train_classifier.py

The three folders will be discussed in the following section:

#### app
The app folder holds files needed to run the web application. Within the templates folder, there are two html templates, defining the layout and contents of the webpage. 
The run.py-file needs to be run in order to launch the web application. This can be done in a console by typing
python run.py

#### data
In this folder the original raw data is provided (disaster_categories.csv and disaster_messages.csv) along with a python script (process_data.py) that reads in, cleans and stores the data in a database.
It is necessary to run the process_data.py file BEFORE launching the webapp (run.py), as it creates a database file that will be read by the webapp.

#### models
The models folder initially only contains one file called train_classifier.py. 
This python script reads in the messages and categories database and trains a classificaction model with the objective of predicting the labels of new messages. 
When running the code, the script creates a trained model and stores this in the models folder. This step must be done before launching the webapp, as the app accesses the pretrained model.
