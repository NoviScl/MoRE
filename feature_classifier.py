import json
import numpy as np
from tqdm import tqdm 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
import pickle
import os
from sklearn import linear_model
from transformers import GPT2TokenizerFast
gpt_tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
from utils import normalize_answer, compute_f1

np.random.seed(2022)

test_dir = "uniqa_predictions_final/test/"
dev_dir = "uniqa_predictions_final/dev/"
datasets = [
    "nq",
    "triviaqa",
    "squad",
    "hotpotqa",
    "beerqa_3hop",
    "musique",
    "gsm8k",
    "svamp",
    "multiarith",
    "csqa",
    "csqa2",
    "qasc"
]
experts = ["factual", "multihop", "math", "commonsense"]

def answer_extract_textqa(pred):
    prefix = "answer is "
    if prefix in pred:
        idx = pred.rfind(prefix)
        return pred[idx + len(prefix) : ]
    return pred.strip()

def extract_features(predictions, qonly=False, agreement=True):
    print ("#total: ", len(predictions))
    correct = 0
    wrong = 0
    X = []
    Y = []
    for i,pred in enumerate(predictions):
        features = []
        
        if "factual" in pred["expert"]:
            features.append(1)
        else:
            features.append(0)
        
        if "multihop" in pred["expert"]:
            features.append(1)
        else:
            features.append(0) 

        if "math" in pred["expert"]:
            features.append(1)
        else:
            features.append(0)
        
        if "commonsense" in pred["expert"]:
            features.append(1)
        else:
            features.append(0)
        
        qword = pred["question"].lower().split()
        if "what" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "who" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "which" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "when" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "where" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "how" in qword and "much" in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "how" in qword and "many" in qword:
            features.append(1)
        else:
            features.append(0)

        if "how" in qword and "much" not in qword and "many" not in qword:
            features.append(1)
        else:
            features.append(0)
        
        if "why" in qword:
            features.append(1)
        else:
            features.append(0)
        
        features.append(len(gpt_tokenizer.tokenize(pred["question"])))

        ## number of numerical values in question
        qwords = pred["question"].lower().split()
        num_count = 0
        for w in qwords:
            if w.replace('.', '').replace('-', '').isnumeric():
                num_count += 1
        features.append(num_count)

        ## for factual & commonsense experts
        if "factual" in pred["expert"] or "commonsense" in pred["expert"]:
            ## number of numerical values in passages
            passages = " ".join(pred["context"])
            pwords = passages.lower().split()
            num_count = 0
            for w in pwords:
                if w.replace('.', '').replace('-', '').isnumeric():
                    num_count += 1
            features.append(num_count)

            ## overlap between question and passages
            qwords = set(pred["question"].lower().split())
            pwords = set(passages.lower().split())
            features.append(len(qwords.intersection(pwords)))

            ## passage length
            features.append(len(gpt_tokenizer.tokenize(passages)))
        else:
            features.extend([-1] * 3)

        if not qonly:
            features.append(pred["lm_prob"])
            features.append(len(gpt_tokenizer.tokenize(pred["answer"])))
            # features.append(np.mean(pred["token_probs"]))
            # features.append(np.max(pred["token_probs"]))
            # features.append(np.min(pred["token_probs"]))
            # features.append(np.std(pred["token_probs"]))

            if agreement:
                ## frequency of answer
                all_answers = []
                for expert in experts:
                    all_answers.append(normalize_answer(pred["answers_by_expert"][expert]))
                # print (all_answers)
                features.append(all_answers.count(normalize_answer(pred["answer"])))

                ## similarity between factual answer and other experts' answers
                if "factual" in pred["expert"]:
                    for expert in experts:
                        if expert != "factual":
                            sim = compute_f1(normalize_answer(pred["answer"]), normalize_answer(pred["answers_by_expert"][expert]))
                            features.append(sim)
                else:
                    features.extend([-1] * 3)
                
                ## similarity between multihop answer and other experts' answers
                if "multihop" in pred["expert"]:
                    for expert in experts:
                        if expert != "multihop":
                            sim = compute_f1(normalize_answer(pred["answer"]), normalize_answer(pred["answers_by_expert"][expert]))
                            features.append(sim)
                else:
                    features.extend([-1] * 3)
                
                ## similarity between math answer and other experts' answers
                if "math" in pred["expert"]:
                    for expert in experts:
                        if expert != "math":
                            sim = compute_f1(normalize_answer(pred["answer"]), normalize_answer(pred["answers_by_expert"][expert]))
                            features.append(sim)
                else:
                    features.extend([-1] * 3)
                
                ## similarity between math answer and other experts' answers
                if "commonsense" in pred["expert"]:
                    for expert in experts:
                        if expert != "commonsense":
                            sim = compute_f1(normalize_answer(pred["answer"]), normalize_answer(pred["answers_by_expert"][expert]))
                            features.append(sim)
                else:
                    features.extend([-1] * 3)
                
                ## similarity between multihop and math full answers 
                if "multihop" in pred["expert"]:
                    sim = compute_f1(normalize_answer(pred["full_answers_by_expert"]["multihop"]), normalize_answer(pred["answers_by_expert"]["math"]))
                    features.append(sim)
                else:
                    features.extend([-1])
                if "math" in pred["expert"]:
                    sim = compute_f1(normalize_answer(pred["full_answers_by_expert"]["math"]), normalize_answer(pred["answers_by_expert"]["multihop"]))
                    features.append(sim)
                else:
                    features.extend([-1])

            ## number of overlap words between question and answer
            qwords = set(pred["question"].lower().split())
            awords = set(pred["answer"].lower().split())
            features.append(len(qwords.intersection(awords)))

            ## number of numerical values in answer
            awords = pred["answer"].lower().split()
            num_count = 0
            for w in awords:
                if w.replace('.', '').replace('-', '').isnumeric():
                    num_count += 1
            features.append(num_count)

            ## for factual & commonsense experts
            if "factual" in pred["expert"] or "commonsense" in pred["expert"]:
                ## number of numerical values in passages
                passages = " ".join(pred["context"])

                ## overlap between answer and passages
                awords = set(pred["answer"].lower().split())
                pwords = set(passages.lower().split())
                features.append(len(awords.intersection(pwords)))

                ## how many times the answer appears in the passages 
                features.append(passages.lower().count(pred["answer"].lower()))
            else:
                features.extend([-1] * 2)

            ## for math / multihop expert
            if pred["expert"] == "math" or pred["expert"] == "multihop":
                ## length of explanation
                features.append(len(pred["full_answer"].split()))

                ## overlap between question and explanation
                qwords = set(pred["question"].lower().split())
                ewords = set(pred["full_answer"].lower().split())
                features.append(len(qwords.intersection(ewords)))

                ## overlap between answer and explanation
                awords = set(pred["answer"].lower().split())
                ewords = set(pred["full_answer"].lower().split())
                features.append(len(awords.intersection(ewords)))

                ## how many times the answer appears in the explanation
                features.append(pred["full_answer"].lower().count(pred["answer"].lower()))

                ## number of numerical values in explanation
                ewords = pred["full_answer"].lower().split()
                num_count = 0
                for w in ewords:
                    if w.replace('.', '').replace('-', '').isnumeric():
                        num_count += 1
                features.append(num_count)
            else:
                features.extend([-1] * 5)

        if pred["em"] == 1:
            correct += 1
            Y.append(1.)
        else:
            wrong += 1
            Y.append(0.)
        
        X.append(features)

    print ("#correct: ", correct)
    print ("#wrong: ", wrong)
   
    X = np.array(X)
    Y = np.array(Y)
    
    print ("data shape: ", X.shape)

    return X, Y

def fit_and_save(X, Y, model, tol=1e-5, name=""):
    if model == "SGD":
        classifier = linear_model.SGDClassifier(loss='log', penalty='l2', max_iter=1000, tol=tol, verbose=1,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=6)
    elif model == "LR":
        classifier = linear_model.LogisticRegression(C=1.0, max_iter=1000, verbose=2, tol=tol)
    elif model == "kNN":
        classifier = KNeighborsClassifier(n_neighbors=5)
    elif model == "SVM":
        classifier = SVC(kernel="linear", C=0.025, verbose=True, probability=True)
        # classifier = SVC(gamma=2, C=1, verbose=True, probability=True)
    elif model == "DecisionTree":
        classifier = DecisionTreeClassifier()
    elif model == "RandomForest":
        classifier = RandomForestClassifier()
    elif model == "AdaBoost":
        classifier =  AdaBoostClassifier()
    elif model == "MLP":
        classifier = MLPClassifier(verbose=True, early_stopping=True, validation_fraction=0.1, n_iter_no_change=2, tol=1e-4)

    classifier.fit(X, Y)
    train_score = classifier.score(X, Y)

    print ("Acc on training set: {:.3f}".format(train_score))

    with open("feature_classifiers/{}_calibrator{}.pkl".format(model, name), "wb") as f:
        pickle.dump(classifier, f)

def load_and_predict(X, Y, model, name=""):
    with open("feature_classifiers/{}_calibrator{}.pkl".format(model, name), "rb") as f:
        classifier = pickle.load(f)
    
    # print ("coefs: ", classifier.coef_)
    
    # return classifier.predict_proba(X)
    # return classifier.predict(X)
    return classifier.score(X, Y)

def intervalECE(all_scores, all_probs, buckets=10):
    '''
    all_scores: EM scores of all predictions.
    all_probs: confidence scores for all predictions.
    buckets: number of buckets.
    '''
    bucket_probs = [[] for _ in range(buckets)]
    bucket_scores = [[] for _ in range(buckets)]
    for i, prob in enumerate(all_probs):
        for j in range(buckets):
            if prob < float((j+1) / buckets):
                break
        bucket_probs[j].append(prob)
        bucket_scores[j].append(all_scores[i])
    
    per_bucket_confidence = [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in bucket_probs
    ]

    per_bucket_score = [
        np.mean(bucket)
        if len(bucket) > 0 else -1.
        for bucket in bucket_scores
    ]

    bucket_sizes = [
        len(bucket) 
        for bucket in bucket_scores
    ]

    print ("Acc: ", [round(num, 2) for num in per_bucket_score])
    print ("Conf: ", [round(num, 2) for num in per_bucket_confidence])
    print ("Sizes: ", bucket_sizes)

    n_samples = sum(bucket_sizes)
    ece = 0.
    for i in range(len(bucket_sizes)):
        if bucket_sizes[i] > 0:
            delta = abs(per_bucket_score[i] - per_bucket_confidence[i])
            ece += (bucket_sizes[i] / n_samples) * delta
    return ece * 100

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == "__main__":    
    # gather all dev set data 
    qonly = False
    agreement = False
    name = "_no_agreement"

    all_predictions = []
    for dataset in datasets:
        predictions_by_expert = {"factual": [], "multihop": [], "math": [], "commonsense": []}
        answers_by_expert = {"factual": [], "multihop": [], "math": [], "commonsense": []}
        full_answers_by_expert = {"factual": [], "multihop": [], "math": [], "commonsense": []}
        for expert in experts:
            fname = os.path.join(dev_dir, dataset + '_' + expert + ".json")
            with open(fname, "r") as f:
                predictions = json.load(f)
            for i in range(len(predictions)):
                predictions[i]["expert"] = fname.split("/")[-1].split(".")[0].split("_")[-1]
                answers_by_expert[expert].append(predictions[i]["answer"])
                full_answers_by_expert[expert].append(predictions[i]["full_answer"])
            predictions_by_expert[expert] = predictions
        for expert in experts:
            for i in range(len(predictions)):
                predictions_by_expert[expert][i]["answers_by_expert"] = {"factual": answers_by_expert["factual"][i], "multihop": answers_by_expert["multihop"][i], "math": answers_by_expert["math"][i], "commonsense": answers_by_expert["commonsense"][i]}
                predictions_by_expert[expert][i]["full_answers_by_expert"] = {"factual": full_answers_by_expert["factual"][i], "multihop": full_answers_by_expert["multihop"][i], "math": full_answers_by_expert["math"][i], "commonsense": full_answers_by_expert["commonsense"][i]}
        for expert in experts:
            all_predictions.extend(predictions_by_expert[expert])
    print ("#train predictions: ", len(all_predictions))


    train_X, train_Y = extract_features(all_predictions, qonly=qonly, agreement=agreement)

    all_predictions = []
    for dataset in datasets:
        predictions_by_expert = {"factual": [], "multihop": [], "math": [], "commonsense": []}
        answers_by_expert = {"factual": [], "multihop": [], "math": [], "commonsense": []}
        full_answers_by_expert = {"factual": [], "multihop": [], "math": [], "commonsense": []}
        for expert in experts:
            fname = os.path.join(test_dir, dataset + '_' + expert + ".json")
            with open(fname, "r") as f:
                predictions = json.load(f)
            for i in range(len(predictions)):
                predictions[i]["expert"] = fname.split("/")[-1].split(".")[0].split("_")[-1]
                answers_by_expert[expert].append(predictions[i]["answer"])
                full_answers_by_expert[expert].append(predictions[i]["full_answer"])
            predictions_by_expert[expert] = predictions
        for expert in experts:
            for i in range(len(predictions)):
                predictions_by_expert[expert][i]["answers_by_expert"] = {"factual": answers_by_expert["factual"][i], "multihop": answers_by_expert["multihop"][i], "math": answers_by_expert["math"][i], "commonsense": answers_by_expert["commonsense"][i]}
                predictions_by_expert[expert][i]["full_answers_by_expert"] = {"factual": full_answers_by_expert["factual"][i], "multihop": full_answers_by_expert["multihop"][i], "math": full_answers_by_expert["math"][i], "commonsense": full_answers_by_expert["commonsense"][i]}
        for expert in experts:
            all_predictions.extend(predictions_by_expert[expert])
    print ("#test predictions: ", len(all_predictions))
        
    test_X, test_Y = extract_features(all_predictions, qonly=qonly, agreement=agreement)


    for model in ["RandomForest"]:
        print (model)
        fit_and_save(train_X, train_Y, model, name=name)

        test_scores = load_and_predict(test_X, test_Y, model, name=name)
        print ("test score: ", test_scores)



    ## inference and save results on test set
    with open("feature_classifiers/RandomForest_calibrator{}.pkl".format(name), "rb") as f:
        classifier = pickle.load(f)

    all_predictions = {}
    for dataset in datasets:
        predictions_by_expert = {"factual": [], "multihop": [], "math": [], "commonsense": []}
        answers_by_expert = {"factual": [], "multihop": [], "math": [], "commonsense": []}
        full_answers_by_expert = {"factual": [], "multihop": [], "math": [], "commonsense": []}
        for expert in experts:
            fname = os.path.join(test_dir, dataset + '_' + expert + ".json")
            with open(fname, "r") as f:
                predictions = json.load(f)
            for i in range(len(predictions)):
                predictions[i]["expert"] = fname.split("/")[-1].split(".")[0].split("_")[-1]
                answers_by_expert[expert].append(predictions[i]["answer"])
                full_answers_by_expert[expert].append(predictions[i]["full_answer"])
            predictions_by_expert[expert] = predictions
        for expert in experts:
            for i in range(len(predictions)):
                predictions_by_expert[expert][i]["answers_by_expert"] = {"factual": answers_by_expert["factual"][i], "multihop": answers_by_expert["multihop"][i], "math": answers_by_expert["math"][i], "commonsense": answers_by_expert["commonsense"][i]}
                predictions_by_expert[expert][i]["full_answers_by_expert"] = {"factual": full_answers_by_expert["factual"][i], "multihop": full_answers_by_expert["multihop"][i], "math": full_answers_by_expert["math"][i], "commonsense": full_answers_by_expert["commonsense"][i]}
    
            test_X, test_Y = extract_features(predictions_by_expert[expert], qonly=qonly, agreement=agreement)
            predict_Y = classifier.predict_proba(test_X)
            predict_Y = [ls[1] for ls in predict_Y]

            all_predictions[dataset + '_' + expert] = predict_Y
    
    with open("feature_classifiers/RandomForest_calibrator{}_test_set_predictions.json".format(name), "w+") as f:
        json.dump(all_predictions, f, indent=4)

