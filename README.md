This repository is based on DeepSpeed-Chat.
The main changes are in /applications.

The general steps are:
# 1. Data processing in applications/DeepSpeed-Chat/local_data
   Since we have the raw dataset:
   integrate separate zip files into a holistic text file
   -> python unzip_scrip.py

   Filter games with "SHOWDOWN"
   -> python order_change.py

   Filter good players
   By tuning parameters, we can filter players of different levels
   -> python good_players.py

   conduct prompt engineering on filtered dataset
   -> python prompt_engieering3.py

   The final prompt files are with the name started with "prompt_"
   
# 2. Fine-tuning in applications/DeepSpeed-Chat/Training
   Supervised fine-tuning -> reward modeling -> RLHF
   More operations on alternate pre-trained models and hyperparameter tuning can be referred in:
      https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat

Due to our privacy, we have not released our data and model yet.
