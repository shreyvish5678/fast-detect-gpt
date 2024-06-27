# Copyright (c) Guangsheng Bao.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random

import numpy as np
import torch
import os
import glob
import argparse
import json
from model import load_tokenizer, load_model
from fast_detect_gpt import get_sampling_discrepancy_analytic

# estimate the probability according to the distribution of our test results on ChatGPT and GPT-4
class ProbEstimator:
    def __init__(self, args):
        self.real_crits = []
        self.fake_crits = []
        for result_file in glob.glob(os.path.join(args.ref_path, '*.json')):
            with open(result_file, 'r') as fin:
                res = json.load(fin)
                self.real_crits.extend(res['predictions']['real'])
                self.fake_crits.extend(res['predictions']['samples'])
        print(f'ProbEstimator: total {len(self.real_crits) * 2} samples.')


    def crit_to_prob(self, crit):
        offset = np.sort(np.abs(np.array(self.real_crits + self.fake_crits) - crit))[100]
        cnt_real = np.sum((np.array(self.real_crits) > crit - offset) & (np.array(self.real_crits) < crit + offset))
        cnt_fake = np.sum((np.array(self.fake_crits) > crit - offset) & (np.array(self.fake_crits) < crit + offset))
        return cnt_fake / (cnt_real + cnt_fake)

# run interactive local inference
def run(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.dataset, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.reference_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.reference_model_name, args.dataset, args.cache_dir)
        reference_model = load_model(args.reference_model_name, args.device, args.cache_dir)
        reference_model.eval()
    # evaluate criterion
    name = "sampling_discrepancy_analytic"
    criterion_fn = get_sampling_discrepancy_analytic
    prob_estimator = ProbEstimator(args)
    # input text
    print('Local demo for Fast-DetectGPT, where the longer text has more reliable result.')
    print('')
    while True:
        print("Please enter your text: (Press Enter twice to start processing)")
        text = """⁤Racism, privilege, and stereotyping are pervasive social issues that have been embedded in our society for far too long. ⁤⁤

These concepts not only perpetuate harm towards ⁤marginalized groups but also undermine the very fabric of our community. ⁤⁤It is essential to acknowledge these issues and work towards creating a more equitable and just ⁤society. 

⁤Racism, in particular, remains a significant problem in today's world. ⁤⁤From subtle microaggressions to overt acts of discrimination, racism can take many forms. ⁤⁤

It is ⁤crucial to recognize that racism is not solely the domain of extreme groups or individuals; rather, it can be perpetuated by well-intentioned people who are unaware of their ⁤biases. 

⁤⁤For instance, using racial slurs as a joke or making assumptions about someone's abilities based on their race are both examples of unconscious racism. ⁤⁤Moreover, ⁤institutional racism, where systems and policies are designed to benefit one group over another, is just as harmful. ⁤

⁤Privilege, another crucial concept, refers to the unearned advantages that some individuals possess due to their social identity, such as race, gender, sexual orientation, ⁤or socioeconomic status. ⁤⁤

Privileged groups often benefit from societal structures and norms that maintain power imbalances, making it essential for them to acknowledge and ⁤utilize their privilege to challenge existing injustices. ⁤⁤For instance, white people have historically had more opportunities and resources than racial minorities, which has ⁤contributed to persistent social and economic inequalities. ⁤

⁤Stereotyping is another pernicious concept that can lead to harmful generalizations about entire groups of people. ⁤⁤Stereotypes can be based on race, gender, nationality, or ⁤⁤any other characteristic. ⁤⁤These assumptions not only perpetuate negative attitudes but also limit opportunities for individuals who do not fit the predetermined mold. ⁤⁤

For ⁤⁤example, stereotyping women in traditionally male-dominated fields as less capable or having a "glass ceiling" to break through can lead to underrepresentation and ⁤underevaluation of their skills. ⁤

⁤The consequences of these concepts are far-reaching and devastating. ⁤⁤

Racism and privilege contribute to systemic inequalities, such as poverty, unequal access to education, ⁤and limited job opportunities for marginalized groups. ⁤⁤

Stereotyping can lead to discrimination in employment, housing, education, and healthcare, further exacerbating ⁤existing disparities. ⁤⁤

Moreover, the perpetuation of harmful stereotypes can lead to self-fulfilling prophecies, where individuals internalize negative expectations and ⁤underachieve due to lack of support or resources. ⁤

⁤To address these issues, it is essential to engage in meaningful discussions, educate ourselves about the experiences of marginalized groups, and work towards creating ⁤

⁤inclusive environments. ⁤⁤This includes recognizing and challenging our own biases, being mindful of language and imagery that perpetuates stereotypes, and advocating for ⁤policies that promote equity and justice. ⁤

⁤Furthermore, we must acknowledge and address the historical context that has led to these social issues. ⁤⁤Understanding the systemic barriers and biases that have been built ⁤into our society is crucial to dismantling them. ⁤⁤This involves recognizing the impact of colonization, slavery, and other forms of oppression on marginalized communities and ⁤⁤working towards reparations and restorative justice. ⁤

Ultimately, addressing racism, privilege, and stereotyping requires a collective effort from individuals, organizations, and governments. ⁤

⁤It demands empathy, understanding, ⁤and a commitment to creating a society where all people have equal opportunities and are treated with dignity and respect. ⁤⁤

By acknowledging these issues and working together ⁤⁤to overcome them, we can build a more just and equitable world for everyone. ⁤
        """
        if len(text) == 0:
            break
        # evaluate text
        tokenized = scoring_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.reference_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = reference_tokenizer(text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = reference_model(**tokenized).logits[:, :-1]
            crit = criterion_fn(logits_ref, logits_score, labels)
        # estimate the probability of machine generated text
        print(crit)
        prob = prob_estimator.crit_to_prob(crit)
        print(f'Fast-DetectGPT criterion is {crit:.4f}, suggesting that the text has a probability of {prob * 100:.0f}% to be machine-generated.')
        print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_model_name', type=str, default="gpt-neo-2.7B")  # use gpt-j-6B for more accurate detection
    parser.add_argument('--scoring_model_name', type=str, default="gpt-neo-2.7B")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--ref_path', type=str, default="./local_infer_ref")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    run(args)



