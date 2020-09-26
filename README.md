# Detect genre of music based on time series music features data

### Highlights:
- Trained a hidden markov models on features data for various genres of music to classify genre of music given music .wav file
- Achieved 0.66 F1   
- Tech: Tensorflow (Keras), Python (Numpy, Pandas, matplotlib)
- Work completed: Data processing (extracting MFCC and Filterbank features from music), Model training (Gaussian HMM, GMM HMM, Multinomial HMM), Model evaluation (F1, confusion matrix)


![Alt text](/assets/results.png?raw=true=50x50  "Forecasting results on test dataset")

### Dataset:
- MARSYAS  https://github.com/marsyas/marsyas

### Hidden Markov Models:
- Hidden markov models are statistical Markov models, where the system being modelled is assumed to be a markov process with hidden states. Only the final state is observed, and all the other internal states are hidden and unobserved.
- The markov proces is a stochastic (randomly determined) process in which future state solely depends on current state only. This helps solve complex ML and RL problems. 
- The markov matrix is a transition matrix that contains the probaility of transitioning from one state to another
- To apply hidden markov models for timeseries classification, we can develop a hidden markov model trained on time series data from each class. This allows us to understand the distribution of the samples that belong to the class. 
- Then given unseen timeseries data, we can calculate the log likelihood of the data belonging to each class. The model that yields the highest likelihood is chosen to be the predicted class for that data. 

### References:
- This project was inspired by the following resources:

-https://www.tutorialspoint.com/artificial_intelligence_with_python/artificial_intelligence_with_python_analyzing_time_series_data.htm

-https://blog.goodaudience.com/music-genre-classification-using-hidden-markov-models-4a7f14eb0fd4
