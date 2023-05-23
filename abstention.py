import numpy as np
import sklearn.metrics as metrics
import json
import os

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
choices = ["A", "B", "C", "D"]

# classifier_name = "_no_agreement"
classifier_name = ""
classifier_scores_path = "/fs/nexus-scratch/noviscl/MoPE/feature_classifiers/RandomForest_calibrator{}_predictions.json".format(classifier_name)
classifier_scores_path_dev = "/fs/nexus-scratch/noviscl/MoPE/feature_classifiers/RandomForest_calibrator{}_dev_set_predictions.json".format(classifier_name)


def get_risk_coverage_info(prob_list, em_list):
    tuples = [(x,y) for x,y in zip(prob_list, em_list)]
    sorted_tuples = sorted(tuples, key=lambda x: -x[0])
    sorted_probs = [x[0] for x in sorted_tuples]
    sorted_em = [x[1] for x in sorted_tuples]
    total_questions = len(sorted_em)
    total_correct = 0
    covered = 0
    risks = []
    coverages = []

    for em, prob in zip(sorted_em, sorted_probs):
        covered += 1
        if em:
            total_correct += 1
        risks.append(1 - (total_correct/covered))
        coverages.append(covered/total_questions)        
    auc = round(metrics.auc(coverages, risks), 4)
    
    return risks, coverages, auc

def cov_at_acc(prob_list, em_list, acc=0.9):
    tuples = [(x, y) for x, y in zip(prob_list, em_list)]
    sorted_tuples = sorted(tuples, key=lambda x: -x[0])
    sorted_em = [x[1] for x in sorted_tuples]
    total_questions = len(sorted_em)
    total_correct = 0
    covered = 0
    coverages = {}

    for em in sorted_em:
        covered += 1
        if em:
            total_correct += 1
        accuracy = total_correct / covered
        coverage = covered / total_questions
        coverages[accuracy] = coverage

    max_coverage = max([coverage for accuracy, coverage in coverages.items() if accuracy >= acc], default=None)
    return max_coverage

def effective_reliability(prob_list, em_list, threshold=0.1):
    er = 0
    for i in range(len(prob_list)):
        if prob_list[i] >= threshold:
            if em_list[i]:
                er += 1
            else:
                er -= 1
    return er / len(prob_list)

def search_er_threshold(prob_list, em_list):
    ## search the best threshold based on the dev set 
    best_er = -100
    best_threshold = -1
    for threshold in np.arange(0.01, 1.0, 0.01):
        er = effective_reliability(prob_list, em_list, threshold=threshold)
        if er > best_er:
            best_er = er
            best_threshold = threshold
    return best_threshold, best_er

def ER_metric(method="maxprob"):
    with open(classifier_scores_path_dev, 'r') as f:
        classifier_scores_dev = json.load(f)
    
    with open(classifier_scores_path, 'r') as f:
        classifier_scores_test = json.load(f)

    all_metrics = []
    prob_list = []
    em_list = []
    ## search for the best theshold on the combined dev set 
    for dataset in datasets:
        all_experts = {}
        all_scores = {}
        for expert in experts:
            fname = os.path.join(dev_dir, dataset + '_' + expert + ".json")
            with open(fname, 'r') as f:
                data = json.load(f)
                all_experts[expert] = data
                all_scores[expert] = classifier_scores_dev[dataset + '_' + expert]

        ## check all lengths are equal 
        if len(set([len(all_experts[experts[0]]), len(all_experts[experts[1]]), len(all_experts[experts[2]]), len(all_experts[experts[3]])])) != 1:
            print ("Lengths are not equal")
            exit()
        for i in range(len(all_experts[experts[0]])):
            maxid = np.argmax([all_scores[experts[0]][i], all_scores[experts[1]][i], all_scores[experts[2]][i], all_scores[experts[3]][i]])
            em = all_experts[experts[maxid]][i]['em']
            em_list.append(em)
            if method == "maxprob":
                prob_list.append(all_experts[experts[maxid]][i]['lm_prob'])
            elif method == "mope":
                prob_list.append(all_scores[experts[maxid]][i])
    best_threshold, best_er = search_er_threshold(prob_list, em_list)
    print ('best dev threshold: ', best_threshold, 'best dev ER: ', best_er)

    for dataset in datasets:
        all_experts = {}
        all_scores = {}
        prob_list = []
        em_list = []
        for expert in experts:
            fname = os.path.join(test_dir, dataset + '_' + expert + ".json")
            with open(fname, 'r') as f:
                data = json.load(f)
                all_experts[expert] = data
                all_scores[expert] = classifier_scores_test[dataset + '_' + expert]

        ## check all lengths are equal 
        if len(set([len(all_experts[experts[0]]), len(all_experts[experts[1]]), len(all_experts[experts[2]]), len(all_experts[experts[3]])])) != 1:
            print ("Lengths are not equal")
            exit()
        for i in range(len(all_experts[experts[0]])):
            maxid = np.argmax([all_scores[experts[0]][i], all_scores[experts[1]][i], all_scores[experts[2]][i], all_scores[experts[3]][i]])
            em = all_experts[experts[maxid]][i]['em']
            em_list.append(em)
            if method == "maxprob":
                prob_list.append(all_experts[experts[maxid]][i]['lm_prob'])
            elif method == "mope":
                prob_list.append(all_scores[experts[maxid]][i])
        ER = effective_reliability(prob_list, em_list, threshold=best_threshold)
        print (dataset, 'test ER: ', ER * 100)
        all_metrics.append(ER)
    print ('Average ER: ', np.mean(all_metrics) * 100)

def all_metric(method="maxprob", metric="AUC"):
    with open(classifier_scores_path, 'r') as f:
        classifier_scores = json.load(f)

    all_metrics = []
    for dataset in datasets:
        all_experts = {}
        all_scores = {}
        prob_list = []
        em_list = []
        for expert in experts:
            fname = os.path.join(test_dir, dataset + '_' + expert + ".json")
            with open(fname, 'r') as f:
                data = json.load(f)
                all_experts[expert] = data
                all_scores[expert] = classifier_scores[dataset + '_' + expert]

        ## check all lengths are equal 
        if len(set([len(all_experts[experts[0]]), len(all_experts[experts[1]]), len(all_experts[experts[2]]), len(all_experts[experts[3]])])) != 1:
            print ("Lengths are not equal")
            exit()
        for i in range(len(all_experts[experts[0]])):
            maxid = np.argmax([all_scores[experts[0]][i], all_scores[experts[1]][i], all_scores[experts[2]][i], all_scores[experts[3]][i]])
            em = all_experts[experts[maxid]][i]['em']
            em_list.append(em)
            if method == "maxprob":
                prob_list.append(all_experts[experts[maxid]][i]['lm_prob'])
            elif method == "mope":
                prob_list.append(all_scores[experts[maxid]][i])
        if metric == "AUC":
            auc = get_risk_coverage_info(prob_list, em_list)[-1]
            print (dataset, auc * 100)
            all_metrics.append(auc)
        elif metric == "Cov@80":
            cov = cov_at_acc(prob_list, em_list, acc=0.8)
            if cov is None:
                cov = 0.0
            print (dataset, cov * 100)
            all_metrics.append(cov)
        elif metric == "Cov@90":
            cov = cov_at_acc(prob_list, em_list, acc=0.9)
            if cov is None:
                cov = 0.0
            print (dataset, cov * 100)
            all_metrics.append(cov)
    if metric == "AUC":
        print ("Average AUC: ", round(np.mean(all_metrics) * 100, 4))
    elif metric == "Cov@80":
        print ("Average Cov@80: ", round(np.mean(all_metrics) * 100, 4))
    elif metric == "Cov@90":
        print ("Average Cov@90: ", round(np.mean(all_metrics) * 100, 4))

if __name__ == "__main__":
    # ER_metric(method="mope")
    all_metric(method="mope", metric="Cov@90")