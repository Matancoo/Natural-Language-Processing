import torch
from transformers import AutoModelForSequenceClassification
import plotly.express as px

from sklearn.model_selection import train_test_split
###################################################
# Exercise 5 - Natural Language Processing 67658  #
###################################################

import numpy as np
import datasets
import sklearn
from transformers import Trainer
from transformers import TrainingArguments

# subset of categories that we will use
category_dict = {'comp.graphics': 'computer graphics',
                 'rec.sport.baseball': 'baseball',
                 'sci.electronics': 'science, electronics',
                 'talk.politics.guns': 'politics, guns'
                 }


def get_data(categories=None, portion=1., sklearn=None):
    """
    Get data for given categories and portion
    :param portion: portion of the data to use
    :return:
    """
    # get data
    from sklearn.datasets import fetch_20newsgroups
    data_train = fetch_20newsgroups(categories=categories, subset='train', remove=('headers', 'footers', 'quotes'),
                                    random_state=21)
    data_test = fetch_20newsgroups(categories=categories, subset='test', remove=('headers', 'footers', 'quotes'),
                                   random_state=21)

    # train
    train_len = int(portion * len(data_train.data))
    x_train = np.array(data_train.data[:train_len])
    y_train = data_train.target[:train_len]
    # remove empty entries
    non_empty = x_train != ""
    x_train, y_train = x_train[non_empty].tolist(), y_train[non_empty].tolist()

    # test
    x_test = np.array(data_test.data)
    y_test = data_test.target
    non_empty = np.array(x_test) != ""
    x_test, y_test = x_test[non_empty].tolist(), y_test[non_empty].tolist()
    return x_train, y_train, x_test, y_test


# Q1
def linear_classification(portion=1.):
    """
    Perform linear classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    tf = TfidfVectorizer(stop_words='english',
                         max_features=1000)  # NOTE: features = number of words we give value to for each given document
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Add your code here
    train_matrix = tf.fit_transform(
        x_train)  # Fit and transform the data, transform returns a sparse matrix of TF-IDF features.
    test_matrix = tf.transform(x_test)  # transform the data, transform returns a sparse matrix of TF-IDF features.
    linear_model = LogisticRegression()

    linear_model.fit(train_matrix, y_train)
    accuracy = linear_model.score(test_matrix, y_test)
    return accuracy


# Q2
def transformer_classification(portion=1.):
    """
    Transformer fine-tuning.
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    import torch
    class Dataset(torch.utils.data.Dataset):
        """
        Dataset object
        """

        def __init__(self, encodings, labels):
            self.encodings = encodings  # TODO: we inpute the entire tokenizer applied on test/train set)
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)

    from datasets import load_metric
    metric = load_metric("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    from transformers import Trainer, TrainingArguments
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('distilroberta-base', cache_dir=None)
    model = AutoModelForSequenceClassification.from_pretrained('distilroberta-base',
                                                               cache_dir=None,
                                                               num_labels=len(category_dict),
                                                               problem_type="single_label_classification")

    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)

    # Add your code here
    # see https://huggingface.co/docs/transformers/v4.25.1/en/quicktour#trainer-a-pytorch-optimized-training-loop
    # Use the DataSet object defined above. No need for a DataCollator

    # call tokenizer on x_train AND build dataset with encodings
    train_batch = tokenizer(x_train, padding='longest', truncation=True)  # of all train set (paragraphs)
    train_dataset = Dataset(train_batch, y_train)

    # call tokenizer on x_test AND build dataset with encodings
    test_batch = tokenizer(x_test, padding='longest', truncation=True)  # of all train set (paragraphs)
    test_dataset = Dataset(test_batch, y_test)

    training_args = TrainingArguments(
        output_dir="path/to/save/folder/",
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        num_train_epochs=5,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics)

    trainer.train()
    save_directory = "saved"
    tokenizer.save_pretrained(save_directory)
    model.save_pretrained(save_directory)

    accuracy = trainer.evaluate()

    return accuracy['eval_accuracy']


# Q3
def zeroshot_classification(portion=1.):
    """
    Perform zero-shot classification
    :param portion: portion of the data to use
    :return: classification accuracy
    """
    from transformers import pipeline
    from sklearn.metrics import accuracy_score
    import torch
    x_train, y_train, x_test, y_test = get_data(categories=category_dict.keys(), portion=portion)
    clf = pipeline("zero-shot-classification",
                   model='cross-encoder/nli-MiniLM2-L6-H768',
                   device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    candidate_labels = list(category_dict.values())
    # Add your code here
    # see https://huggingface.co/docs/transformers/v4.25.1/en/main_classes/pipelines#transformers.ZeroShotClassificationPipeline
    results = clf(x_test, candidate_labels)
    labels_dic = {label:i for i,label in enumerate(candidate_labels)}

    preds = [labels_dic[result["labels"][0]] for result in results]
    accuracy = accuracy_score(y_test, preds)
    return accuracy


def plot_accuracy(accuracies, portions, title):
    fig = px.scatter(x=portions, y=accuracies, size=np.ones(len(portions)),
                     size_max=10,
                     title=title)
    fig.show()





if __name__ == "__main__":

    portions = [0.1, 0.5, 1.]
    accuracies1 = []
    accuracies2 = []
    # Q1
    print("Logistic regression results:")
    for p in portions:
        print(f"Portion: {p}")
        accuracy = linear_classification(p)
        print(accuracy)
        accuracies1.append(accuracy)
    plot_accuracy(accuracies1, portions,"Logistic Regression Accuracy per portion of Data")

    # # # Q2
    print("\nFinetuning results:")
    for p in portions:
        print(f"Portion: {p}")
        accuracy = transformer_classification(portion=p)
        accuracies2.append(accuracy)
    for i in range(len(portions)):
        print(f"Portion: {portions[i]}")
        print(accuracies2[i])
    plot_accuracy(accuracies2, portions, "model accuracy per portion of dataset")

    # # Q3
    print("\nZero-shot result:")
    print(zeroshot_classification())
