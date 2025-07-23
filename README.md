
## setup

please use: Python .10.14

install required packages through the requirements.txt file.

## Pre processing

make multimodal arousal detection using:
https://github.com/abrinkk/multimodal-arousal-detector

place them in an appripriate arousal path

RUN:
python sleep_preprocessing.py --input_csv /path/to/records.csv --arousal_path /path/to/arousals/ --target_path /path/to/output/

## training script
train a model using the main script
RUN:
python main.py

## runnig a saved model
run a saved model using the runsaved script
RUN:
python runsaved_240.py --data_input /path/to/info_file.csv --test_size 500 --model_path /path/to/model.pth --output_path /path/to/results.csv


## Apnotyping
to see the apnotypes for each event
RUN:
python apnotyping.py --data_input /path/to/info_file.csv --model_path /path/to/model.pth --output_dir /path/to/output/
