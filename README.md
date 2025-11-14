🚀 Project Overview

<br>

This project demonstrates how to build a spam classifier using natural language processing (NLP) and neural networks.

The workflow includes:

Dataset loading & exploration

Text cleaning and preprocessing

Tokenization & sequence padding

Word cloud for visualization

Train-test split

Deep learning model with Embedding + LSTM

Model training with callbacks

Evaluation using accuracy, loss curves, and predictions

<br>

📂 Dataset

The notebook uses the spam_ham_dataset.csv dataset.
Each entry contains:

text → the SMS message

label → spam or ham

Before training, labels are converted to binary format (spam = 1, ham = 0).

<br>

🛠️ Technologies Used

Python

Pandas, NumPy

NLTK (stopwords, regex)

Matplotlib / Seaborn

WordCloud

TensorFlow / Keras

Scikit-learn

<br>

🔧 Steps Performed
1. Importing Libraries

Essential NLP, visualization, and deep learning libraries.

2. Loading & Understanding the Dataset

View head of the data

Inspect structure using .info(), .shape, .columns

3. Balancing the Dataset

Ensures equal representation of spam and ham messages for fair training.

4. Text Preprocessing

Text cleanup includes:

Lowercasing

Removing punctuation/special characters

Removing stopwords

Regex-based cleanup

5. Visualization

Word clouds are generated for:

Spam messages

Ham messages

This helps understand frequently appearing words.

6. Tokenization & Sequence Padding

Convert text into tokens using KerasTokenizer

Pad sequences to equal length

Prepare labels (train_Y, test_Y)

7. Model Architecture

A Sequential neural network with:

Embedding Layer

LSTM Layer

Dense Output Layer with Sigmoid activation

Compiled with:

Binary Crossentropy loss

Adam optimizer

Accuracy metric

8. Training the Model

Callbacks used:

EarlyStopping – prevent overfitting

ReduceLROnPlateau – dynamic learning rate adjustment

9. Model Evaluation

Training vs. validation accuracy

Training vs. validation loss

Final accuracy on test data

10. Predictions

Model predicts whether a given message is Spam (1) or Ham (0).

<br>

📊 Results

The model achieves strong performance with high validation accuracy, demonstrating effective text understanding through LSTM layers.
(You can share exact accuracy if you'd like it included.)

<br>

📁 Repository Structure
📦 Spam-Classifier
 📜 Spam_Classif.ipynb
 📜 README.md
 📜 requirements.txt (optional)
 📁 dataset/ (optional)

🤝 Contribution

Feel free to raise issues or submit pull requests to improve the project!

⭐ If you find this useful

Consider giving the repository a star to support the project!
