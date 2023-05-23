import os
import numpy as np
from time import sleep
import json
import random
from utils import normalize_answer

random.seed(2023)

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

def find_most_common_index(lst):
    counts = {}  # Dictionary to store the count of each element

    # Count the occurrences of each element in the list
    for i, element in enumerate(lst):
        if element in counts:
            counts[element] += 1
        else:
            counts[element] = 1

    # Find the most common element and its index
    most_common = max(counts, key=counts.get)
    if counts[most_common] == 1:
        most_common_index = random.randint(0, 3)
    else:
        most_common_index = lst.index(most_common)

    return most_common_index

def main():
    
    all_gpt_preds = {}
    for dataset in datasets:
        correct = 0
        total = 0
        all_dp = {"dataset": dataset, "data": []}
        all_data = {}
        for expert in experts:
            fname = os.path.join(test_dir, dataset + '_' + expert + ".json")
            with open(fname, 'r') as f:
                data = json.load(f)
            all_data[expert] = data
        
        for i in range(len(all_data["factual"])):
            total += 1
            EMs = []
            dp = {}
            dp["question"] = all_data["factual"][i]["question"].strip()
            
            EMs.append(all_data["factual"][i]["em"])
            dp["factual"] = normalize_answer(all_data["factual"][i]["answer"].replace("\n", " ").strip())
            
            EMs.append(all_data["multihop"][i]["em"])
            dp["multihop"] = normalize_answer(all_data["multihop"][i]["answer"].replace("\n", " ").strip())

            EMs.append(all_data["math"][i]["em"])
            dp["math"] = normalize_answer(all_data["math"][i]["answer"].replace("\n", " ").strip())
            
            EMs.append(all_data["commonsense"][i]["em"])
            dp["commonsense"] = normalize_answer(all_data["commonsense"][i]["answer"].replace("\n", " ").strip())
            dp["EMs"] = EMs
            
            ## random selection
            # output = random.choice(choices)

            ## majority vote 
            output = choices[find_most_common_index(EMs)]

            dp["choice"] = output
            dp["EM"] = EMs[choices.index(output)]
            correct += EMs[choices.index(output)]
        
            all_dp["data"].append(dp)

        accuracy = correct / total * 100 
        print ("Accuracy on {}: {} / {} = {}%".format(dataset, correct, total, accuracy))
        all_gpt_preds[dataset] = all_dp

    # with open("gpt_preds.json", 'w') as f:
    #     json.dump(all_gpt_preds, f, indent=4)

if __name__ == '__main__':
    main()