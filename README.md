# Some ML project notebooks:

# Client segmentation with clustering and visualization

The goal was to create a client segmentation pipeline based on a Brazilian online retailer data. The first part of the project was the cleaning of the data and feature engineering based on client behavior such as spending patterns, frequency of purchases. In a second time, we used the transformed data to produce the segmentation based on two approaches, a classic scoring-based segmentation, then an unsupervised learning approach using sklearn built-in models such as k-means, dbscan, optics, evaluating the performance of the different algorithms on the data. Finally we produced visualizations of the segmentation for most of the strategies.  

# Computer vision project

This project was an image classification project based on the Stanford dog dataset with over 50 species. I used data augmentation to improve the dataset. The Keras framework was then the main tool for the ML pipeline. The project tried to compare various strategies notably the improvement from a base model and several transfer learning models such as InceptionV3 and VGG16. Despite good results, it was clear that the initial dataset probably needed reviewing because of some mislabelling (Google image results told us that some species were not correct) and a tendency in the dataset to distinguish between extremely similar looking dog subspecies which the algorithm (and humans!) could not distinguish. 

# Energy usage prediction

This project is a classic regression/prediction project. First part was an EDA and some data cleaning, feature engineering like categorical encoding, log transform. Then we applied sklearn models for tabular data such as linear regularization, random forests or xgboost to the various prediction targets. Finally, we optimized the hyperparameters of the top performing models. 

# NLP classification project

In this NLP project, we tried to apply both supervised and unsupervised learning to a text corpus taken from StackOverflow SQL database that gives tagged forum posts. We had a rather high amount of tags and every post could have multiple at a time. First, base approach to classify forum posts both supervised and unsupervised ML for sparse multitag classification.Topic-modelling with tfidf and LDA and embedding strategies with gensim were used to extract ranked tags from each text. We also used a supervised approach using RFs and XGBoost for multilabel classification. 

We hit difficulties mainly due to the sparsitiy of the labels and the nature of the tags provided by stack overflow which were not specific enough to train the models properly.

In a second time, we tried to go back to the project using more recent transformer algorithms such as Bert-based models and XLnet for supervised ML. We saw a huge jump in performance using those attention-based models even with a reduced training set. The main drawback was obviously the computational cost. 
