# PhilosophyLSTM
Can a LSTM philosophise? Answer: Not really haha. Training a LSTM on Plato's "The Republic". Inspired after reading Andrej Karpathy's arcticle on LSTMs http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 

# Notes

 - Run on GPU, otherwise it takes ages
 - More epochs when training the LSTM => more congruent sentences produced (only tried up to 30 epochs, AWS is expensive hehe)
 - Hyperparemeters of Word2Vec and LSTM were not really tuned since that would be quite costly (I used parameters/structure that were common in literature)
 - Pro Tip: I used AWS, but Google Cloud is cheaper :P 

# Running

First run

```
create_word2vec.py
```

this will create the word2vec vectors for the text, which will be stored in vectors.bin. Then run:

```
lstm_trainer_generator.py
```

To both train the LSTM and then generate sample sentences. If you want only to train the LSTM use the method train_model(), this method saves the weights to a file called "lstm-weights". If you only want to generate sentences, you need an already pre-trained LSTM (i.e. the weights file) and then use generate_sentences() method. To change the structure of the LSTM edit the file:

```
lstm_model.py
```
