import json 
import numpy as np
import os 
from utils import normalize_answer
from tqdm import tqdm

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
dataset_to_expert_map = {
    "nq": 0,
    "triviaqa": 0,
    "squad": 0,
    "hotpotqa": 1,
    "beerqa_3hop": 1,
    "musique": 1,
    "gsm8k": 2,
    "svamp": 2,
    "multiarith": 2,
    "csqa": 3,
    "csqa2": 3,
    "qasc": 3
}

classifier_scores_path = "/fs/nexus-scratch/noviscl/MoPE/feature_classifiers/RandomForest_calibrator_predictions.json"

def ensemble(data_dir = test_dir, method = "oracle", temp_a = 1.0, temp_b = 1.0, temp_c = 1.0, temp_d = 1.0):
    '''
    oracle: take the max EM among all experts
    question-oracle: assign each question to the corresponding expert 
    maxprob: pick the max prob answer among all experts
    classifer: use the classifier scores to ensemble
    '''
    if method == "classifier":
        with open(classifier_scores_path, 'r') as f:
            classifier_scores = json.load(f)

    all_acc = []
    for dataset in datasets:
        # print (dataset)
        scores = []
        all_experts = {}
        all_scores = {}
        for expert in experts:
            df = os.path.join(data_dir, dataset + '_' + expert + ".json")
            with open(df, 'r') as f:
                data = json.load(f)
                all_experts[expert] = data
                all_scores[expert] = classifier_scores[dataset + '_' + expert]
        ## check all lengths are equal 
        if len(set([len(all_experts[experts[0]]), len(all_experts[experts[1]]), len(all_experts[experts[2]]), len(all_experts[experts[3]])])) != 1:
            print ("Lengths are not equal")
            exit()
        for i in range(len(all_experts[experts[0]])):
            ## check all questions are equal
            if normalize_answer(all_experts[experts[0]][i]['question']) != normalize_answer(all_experts[experts[1]][i]['question']) or normalize_answer(all_experts[experts[0]][i]['question']) != normalize_answer(all_experts[experts[2]][i]['question']) or normalize_answer(all_experts[experts[0]][i]['question']) != normalize_answer(all_experts[experts[3]][i]['question']):
                print ("Questions are not equal")
                exit()
            if method == "oracle":
                score = max(all_experts[experts[0]][i]['em'], all_experts[experts[1]][i]['em'], all_experts[experts[2]][i]['em'], all_experts[experts[3]][i]['em'])
            elif method == "question-oracle":
                maxid = dataset_to_expert_map[dataset]
                score = all_experts[experts[maxid]][i]['em']
            elif method == "maxprob":
                maxid = np.argmax([all_experts[experts[0]][i]['lm_prob'] * temp_a, all_experts[experts[1]][i]['lm_prob'] * temp_b, all_experts[experts[2]][i]['lm_prob'] * temp_c, all_experts[experts[3]][i]['lm_prob'] * temp_d])
                score = all_experts[experts[maxid]][i]['em']
            elif method == "classifier":
                maxid = np.argmax([all_scores[experts[0]][i], all_scores[experts[1]][i], all_scores[experts[2]][i], all_scores[experts[3]][i]])
                score = all_experts[experts[maxid]][i]['em']
            scores.append(score)

        # print ("Average EM: {} / {} = {}% ".format(np.sum(scores), len(scores), np.mean(scores)*100))
        all_acc.append(np.mean(scores)*100)
    return all_acc

## optimal: 8, 5, 8, 5
def re_scale():
    ## tune for an optimal set of temperature values to rescale prob of each expert 
    best_acc = 0
    best_a = 0
    best_b = 0
    best_c = 0
    best_d = 0
    for temp_a in range(1, 10):
        for temp_b in range(1, 10):
            for temp_c in range(1, 10):
                for temp_d in range(1, 10):
                    all_acc = ensemble(data_dir = dev_dir, method = "maxprob", temp_a = temp_a, temp_b = temp_b, temp_c = temp_c, temp_d = temp_d)
                    print ("current: ", temp_a, temp_b, temp_c, temp_d)
                    if np.mean(all_acc) > best_acc:
                        best_acc = np.mean(all_acc)
                        best_a = temp_a
                        best_b = temp_b
                        best_c = temp_c
                        best_d = temp_d
                        print ("Best Acc So Far: {}%".format(best_acc))
                        print ("Best Params So Far: {}, {}, {}, {}".format(best_a, best_b, best_c, best_d))
    print ("Final Best Acc: {}%".format(best_acc))
    print ("Final Best Params: {}, {}, {}, {}".format(best_a, best_b, best_c, best_d))
    return best_a, best_b, best_c, best_d

if __name__ == "__main__":
    all_acc = ensemble(data_dir = test_dir, method = "classifier")
    print ("Test Acc: ", all_acc)
    print ("Average EM: {}% ".format(np.mean(all_acc)))

