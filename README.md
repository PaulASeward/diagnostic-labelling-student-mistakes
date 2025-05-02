## Installation

Ensure you have Python 3.6 or newer installed. It's recommended to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required dependencies:

```bash 
pip install -r requirements.txt
```

If this virtual environment is already created and requirements installed into it, we can just activate to use it:

```bash
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

# Mistake Label Classification

## Prepping the Data

Before running the application clean the data by running the command:
    
```bash
python3 -m mistake_label_classification.clean_utils
```

## Running the Application

Make sure the data to analyze has been placed in the data directory, labelled as "mistake-data.csv" To start the Dash server and run the application as regular, execute:
    
```bash
python3 mistake_label_app.py
```

If the task did NOT already have pre-generated embeddings or categorical hints for the mistake labels, run the app with the following command(s) depending on what is missing:

```bash
python3 mistake_label_app.py --debug=True --generate_embeddings=True --generate_hints=True
```


# Data Visualization of TA Corrections

This project is a Dash web application designed to visualize the corrections made by Teaching Assistants (TAs) in a educational setting. It provides insights into grading patterns, feedback differences, and the overall impact of TAs on student assignments.

## Prepping the Data

Before running the application clean the data by running the command:
    
```bash
python3 -m ta_feedback_analysis.build_ta_overrides_table.py
```

## Running the Application

Make sure the data to analyze has been placed in the data directory, labelled as "mistake-data.csv" To start the Dash server and run the application as regular, execute:
    
```bash
python3 ta_feedback_app.py
```

## Features

- Course, assignment, and task selection via dropdown menus.
- Dynamic scatter plots to visualize TA feedback trends and grading patterns.
- Heatmap visualization of feedback similarity.
- Data-driven insights to enhance teaching quality and assignment grading.


Ensure the dash app is closed before running another. The application will be available at http://127.0.0.1:8050/

