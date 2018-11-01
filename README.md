# chatbot
A chatbot which understands stories and answers questions based on stories using Tensorflow.

Comment out the #train section if you want to use the pre-trained weight.

Run chatbot.py.

The highest correct rate during testing is 98% for single_supporting_fact_10k and 55% for two_supporting_facts_10k.

Configure challenge type by commenting and uncommenting challenge_type.

Run pip install -r requirements.txt or pip3 install -r requirements.txt to install dependencies.

Use dir = os.getcwd() on Jupyter Notebook or dir = os.path.dirname(__file__) on Command Line local runtime.

Used single layer LSTM on single_supporting_fact questions, and two layers LSTM on two_supporting_facts questions.

The optimal amount of epoch for single_supporting_fact questions is 120 and for two_supporting_facts questions is 40 during testing.

Add more dropout layers or increase dropout rate if training for more epochs.


