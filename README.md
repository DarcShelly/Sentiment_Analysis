# Sentiment_Analysis
Sentiment Analysis with CNN
## Project Overview
This repository contains the implementation of a sentiment analysis model using deep learning techniques. Sentiment analysis, also known as opinion mining, involves determining the sentiment or emotion expressed in a piece of text. This project aims to classify text data into positive, negative, or neutral sentiments using state-of-the-art deep learning models.
### What is Sentiment Analysis?
Sentiment analysis is a natural language processing (NLP) task that involves identifying and categorizing opinions expressed in a text. It is widely used in various applications such as:

Social Media Monitoring: Understanding public opinion about products, brands, or events.
Customer Feedback Analysis: Analyzing reviews to gauge customer satisfaction.
Market Research: Assessing consumer sentiment towards competitors or market trends.
Political Analysis: Tracking sentiment around political events or figures.
By extracting sentiments from text data, businesses and researchers can gain valuable insights and make data-driven decisions.
### Role of Deep Learning in Sentiment Analysis
Deep learning has significantly advanced the field of sentiment analysis by providing powerful models that can capture complex patterns in text data. Traditional machine learning methods often require extensive feature engineering and are limited in their ability to handle large-scale data. In contrast, deep learning models, particularly those based on neural networks, can automatically learn features from raw text data, enabling more accurate and scalable sentiment analysis.

Key deep learning techniques include:
Recurrent Neural Networks (RNNs): Suitable for sequential data, RNNs can capture temporal dependencies in text.
Long Short-Term Memory (LSTM) Networks: A type of RNN that can remember long-term dependencies, making them effective for sentiment analysis.
Convolutional Neural Networks (CNNs): Typically used in image processing, CNNs can also be applied to text data to identify local patterns.
Transformers: State-of-the-art models like BERT and GPT, which leverage attention mechanisms to understand context and relationships in text.

In this project, we focus specifically on CNN models for sentiment analysis.

### Database
Any machine learning model is only as good as its database permits it. Choosing a database for our project proved to be a challenging task, as we wished to predict not only the polarity behind the text but also the emotion—such as joy, sadness, neutral, etc. This requirement for a broader prediction range eliminated many of the top text databases available on Kaggle.

We chose to go forward with the dataset [Text with Sentiment](https://www.kaggle.com/datasets/divu2001/text-with-sentiment), which contains unprocessed tweet texts mapped to their sentiments. This dataset provides a rich variety of sentiments, allowing our model to learn and predict a wide range of emotional responses.

### Preprocessing
The raw text data consists of user comments that contain unnecessary and problematic elements such as punctuation, determiners, and various other characters. This raw text is not suitable for direct input into the model. To prepare the text for modeling, we need to perform several preprocessing steps, including trimming words to their base forms, correcting spelling errors, splitting text into tokens, and removing irrelevant words. Additionally, certain emotions like "disgust" and "shame" were found to be too infrequent to significantly contribute to the model's performance, so these were excluded from the dataset.

The preprocessing steps are as follows:
1. Remove extra white spaces and punctuation characters.
2. Filter out meaningless or irrelevant words.
3. Stemming or Lemmatization. [1](https://www.ibm.com/topics/stemming-lemmatization#:~:text=The%20practical%20distinction%20between%20stemming,be%20found%20in%20the%20dictionary.)
4. Exclude instances of infrequent emotions.
5. Tokenize sentences into individual words (tokens).

### Embedding Techniques
After preprocessing, the text was converted into embeddings using the Word2Vec Skip-Gram model. To keep the vocabulary at a manageable level, only words that occur more than once in the dataset were included. These embeddings were then fed into the model for categorical classification in sentiment analysis.

#### Skip-Gram (Word2Vec)
The Skip-Gram model, part of the Word2Vec family, predicts surrounding words (context) given a central word. This method captures the meaning of words based on their context, generating meaningful word vectors. While the Skip-Gram model may take longer to train compared to the Continuous Bag of Words (CBOW) model, the training time was reasonable, taking only about 30 seconds. [2](https://www.baeldung.com/cs/word-embeddings-cbow-vs-skip-gram) 


### Model Architecture
The model used for sentiment analysis is a Convolutional Neural Network (CNN) designed to handle text embeddings. The architecture is built using multiple convolutional layers with varying filter sizes to capture different features from the text data [3](https://doi.org/10.48550/arXiv.2006.03541) . The key components of the model are as follows:

1. **Embedding Layer**: The model starts with an embedding layer, which utilizes pre-trained embeddings to convert words into dense vectors. This layer is non-trainable to leverage the learned word representations effectively.

2. **Convolutional Layers**: Five convolutional layers are applied, each with a different filter size (2, 3, 4, 5, and 6). These layers extract different n-gram features from the input text, helping the model to learn patterns at multiple levels of granularity.

3. **Global Max Pooling**: Each convolutional layer is followed by a Global Max Pooling layer, which reduces the dimensionality by selecting the maximum value from each feature map, capturing the most important feature detected by the filters.

4. **Merge Layer**: The outputs from all the pooling layers are concatenated, combining the features learned from different filter sizes.

5. **Fully Connected Layers**: The merged features are passed through two fully connected (Dense) layers. The first Dense layer has 128 units with ReLU activation and includes dropout regularization to prevent overfitting. The second Dense layer is the output layer, using a softmax activation function to predict the probability distribution over the target emotion classes.

6. **Compilation**: The model is compiled with categorical cross-entropy as the loss function, Adam as the optimizer, and accuracy as the evaluation metric.

This CNN model is designed to efficiently capture and classify the sentiments expressed in the text data. By leveraging multiple convolutional layers with different filter sizes, the model can detect various patterns and effectively categorize emotions in the text.

## Result
n our sentiment analysis project, we evaluated the model's performance on both the training and test datasets. The model achieved an accuracy of 60% on the training data, indicating a good fit to the training set. However, it performed slightly less well on the test data, with an accuracy of 55.45%. This suggests that while the model generalizes reasonably well, there is still room for improvement in its ability to handle unseen data.
