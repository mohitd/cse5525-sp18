We use 6B 300d GloVe embeddings. Download and unzip http://nlp.stanford.edu/data/glove.6B.zip and use glove.6B.100d.txt into the glove folder
Download the lab dataset and put all of the text files under data/
For this project we used the Keras framework, to install run pip install keras or conda install keras depending on your enviroment.

To run the model: 
For the first run you will need to generate the the training and test sets. Once the arrays are complete they are saved,
at this point you can comment out the generation code for future test, just uncomment out the load calls outlined in the 
python file.  The model will begin to run, the Tensorboard will show your progress through the epochs in the interperter window. 

Results: 
For part A we used the glove word embedings with a single LSTM layer -> Max Pooling layer -> Dense Layer. Below are the results:

1-layer LSTM-256 Accuracy: 0.2301
1-layer LSTM-512 Accuracy: 0.3045
1-layer LSTM-1024 Accuracy: 0.2174

The LSTM with a 512 output dimensionality performed much higher than the other models tested. We attempted to combine the two 
top performing LSTM layers to increase performance. Below are the results: 
 
2-layer LSTM-512 LSTM-256 Accuracy: 0.2640

While this did perform better than the 1024 and 256 models, it still performed worse than the single layer 512 model. 
For Part B we attempted to use the Keras Gated Recurrent Unit, we wanted to test the differences between models so we tested 
the GRU model with the same parameters as the top two performing models.

1-layer GRU-512 Accuracy: 0.2861
2-layer GRU-512 GRU-256Accuracy: 0.2405

The GRU models performed slightly worse than their LSTM counterparts, but they had a faster run time. 
Screen shots are available in the screenshot.docx found in the lab 4 folder. 

