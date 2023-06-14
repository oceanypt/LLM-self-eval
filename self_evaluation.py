import sys
import os
from itertools import combinations
from collections import defaultdict
from bert_score import score
import numpy as np
import argparse
from rouge import Rouge 
import random
import tqdm
import warnings
import json
import pandas as pd
import plotly.express as px

warnings.filterwarnings("ignore")

rouge = Rouge()

def F_alpha(scores, beta = 0.2):
    return (1 + beta**2) * scores[0].numpy() * scores[1].numpy() / ( beta**2 * scores[0].numpy() + scores[1].numpy())
    
def F_alpha_2(scores, beta = 0.2):
    return (1 + beta**2) * np.array(scores[0]) * np.array(scores[1]) / ( beta**2 * np.array(scores[0]) + np.array(scores[1]))


def compute_elo(battles, K=4, SCALE=400, BASE=10, INIT_RATING=1000):
    # code copied from: https://colab.research.google.com/drive/1lAQ9cKVErXI1rEYq7hTKNaCQ5Q8TzrI5?usp=sharing
    rating = defaultdict(lambda: INIT_RATING)
 
    for model_a, model_b, win in battles:
        ra = rating[model_a]
        rb = rating[model_b]
        ea = 1 / (1 + BASE ** ((rb - ra) / SCALE))
        eb = 1 / (1 + BASE ** ((ra - rb) / SCALE))
        if win == model_a:
            sa = 1
        elif win == model_b:
            sa = 0
        elif win == "tie":
            sa = 0.5
        else:
            raise Exception(f"unexpected vote {win}")
        rating[model_a] += K * (sa - ea)
        rating[model_b] += K * (1 - sa - eb)
    
    return rating


def predict_win_rate(elo_ratings, SCALE=400, BASE=10, INIT_RATING=1000):
    names = sorted(list(elo_ratings.keys()))
    wins = defaultdict(lambda: defaultdict(lambda: 0))
    for a in names:
        for b in names:
            ea = 1 / (1 + BASE ** ((elo_ratings[b] - elo_ratings[a]) / SCALE))
            wins[a][b] = ea
            wins[b][a] = 1 - ea

    data = {
        a: [wins[a][b] if a != b else np.NAN for b in names]
        for a in names
    }

    df = pd.DataFrame(data, index=names)
    df.index.name = "model_a"
    df.columns.name = "model_b"
    return df.T


def main():
    parser = argparse.ArgumentParser(prog='', description='')
    parser.add_argument('-i,', '--input', required=True, help="folder to model outputs")
    parser.add_argument('-b,', '--beta', type=float, default=0.2, help="the beta parameter when calculating F score. the default is 0.2")
    parser.add_argument('-m,', '--metric',required=True, choices=['rouge', 'bert-score'], help="the evaluation metric for each pair")
    parser.add_argument('-f,', '--model_type', default='xlnet-large-cased' , choices=['roberta-large', 'xlnet-large-cased', 'allenai/longformer-large-4096', 'facebook/bart-large'], help="the evaluation metric for each pair")
    parser.add_argument('-r,', '--reranking', action='store_true', help="reranking by the top model")
    parser.add_argument('-o,', '--output', default='self_evaluation_result.pdf' , help="path to save image of battle results")
    parser.add_argument('-t,', '--toRemove', nargs='+', type=str, help="the models to be removed in evaluation")
    
    


    args = parser.parse_args()


    # Specify the directory you want to start from
    rootDir = args.input

    all_datasets = [] ## filename, data outputs.


    for dirName, subdirList, fileList in os.walk(rootDir):
        print(f'Found directory: {dirName}')
        for fname in fileList:
            print(f'\tFound file: {fname}')
            # Full path to the file
            file_path = os.path.join(dirName, fname)
            with open(file_path, 'r') as f:
                # Now you can perform operations on the file
                content = f.readlines()
                all_datasets.append((fname, content))

    pairwise_comps = [] ## ('model_a', 'model_b', 'win')


    num_wins = {}
    to_remove = {}
    if args.toRemove:
        for t in args.toRemove:
            to_remove[t] = ""
    
        


    cache_scores = {} ## use cache to speed up evaluation for bert-score
    combs = combinations(list(range(len(all_datasets))), 2)
    for comb in combs:
        model_a = all_datasets[comb[0]] # fname, outputs
        model_b = all_datasets[comb[1]] # fname, outputs

        for i in range(len(all_datasets)):
            if i != comb[0] and i != comb[1] and all_datasets[i][0] not in to_remove:
                ## compute the bert score
                hyps = all_datasets[i][1]
                refs_a = model_a[1]
                refs_b = model_b[1]
            
                if args.metric == 'bert-score':
                    ## use cache to speed up evaluation
                    if str(i)+'_'+str(comb[0]) in cache_scores:
                        score_a = cache_scores[str(i)+'_'+str(comb[0]) ]
                    else:
                        score_a = score(hyps, refs_a, lang='en', verbose=True, model_type=args.model_type, batch_size=32) 
                        cache_scores[str(i)+'_'+str(comb[0])] = (score_a[0], score_a[1])
                        cache_scores[str(comb[0])+'_'+str(i)] = (score_a[1], score_a[0]) ### --> fast

                    score_a = F_alpha(score_a, args.beta).tolist()
                    if str(i)+'_'+str(comb[1]) in cache_scores:
                        score_b = cache_scores[str(i)+'_'+str(comb[1]) ]
                    else:
                        score_b = score(hyps, refs_b, lang='en', verbose=True, model_type=args.model_type, batch_size=32)
                        cache_scores[str(i)+'_'+str(comb[1])] = (score_b[0], score_b[1])
                        cache_scores[str(comb[1])+'_'+str(i)] = (score_b[1], score_b[0]) ### --> fast


                    score_b = F_alpha(score_b, args.beta).tolist()
                elif args.metric == 'rouge':
                    if str(i)+'_'+str(comb[0]) in cache_scores:
                        score_a = cache_scores[str(i)+'_'+str(comb[0]) ]
                    else:
                        score_a = rouge.get_scores(hyps, refs_a)
                        score_a = ([sa['rouge-l']['p'] for sa in score_a], [sa['rouge-l']['r'] for sa in score_a])
                        cache_scores[str(i)+'_'+str(comb[0])] = (score_a[0], score_a[1])
                    score_a = F_alpha_2(score_a, args.beta)

                    if str(i)+'_'+str(comb[1]) in cache_scores:
                        score_b = cache_scores[str(i)+'_'+str(comb[1]) ]
                    else:
                        score_b = rouge.get_scores(hyps, refs_b)
                        score_b = ([sb['rouge-l']['p'] for sb in score_b], [sb['rouge-l']['r'] for sb in score_b])
                        cache_scores[str(i)+'_'+str(comb[1])] = (score_b[0], score_b[1])
                    score_b = F_alpha_2(score_b, args.beta)

                    

                c_pattern = model_a[0] + '-' + model_b[0] + '*' + all_datasets[i][0] + '*'           
                for sa, sb in zip(score_a, score_b):
                    if sa > sb:
                        pairwise_comps.append((model_a[0], model_b[0], model_a[0]))

                        if model_a[0] not in num_wins:
                            num_wins[model_a[0]] = 0
                        num_wins[model_a[0]] += 1

                    elif sa < sb:
                        pairwise_comps.append((model_a[0], model_b[0], model_b[0]))
                        if model_b[0] not in num_wins:
                            num_wins[model_b[0]] = 0
                        num_wins[model_b[0]] += 1

                    elif sa == sb:
                        pairwise_comps.append((model_a[0], model_b[0], 'tie'))


    random.shuffle(pairwise_comps)
    win_rate = predict_win_rate(compute_elo(pairwise_comps, K=4, SCALE=400, BASE=10, INIT_RATING=1000))
    ordered_models = win_rate.mean(axis=1).sort_values(ascending=False).index
    fig = px.imshow(win_rate.loc[ordered_models, ordered_models], 
                color_continuous_scale='RdBu', text_auto=".2f",
                title="Predicted Win Rate Using Elo Ratings for Model A in an A vs. B Battle")
    fig.update_layout(xaxis_title="Model B", 
                  yaxis_title="Model A",
                  xaxis_side="top", height=600, width=600,
                  title_y=0.07, title_x=0.5)
    fig.update_traces(hovertemplate=
                  "Model A: %{y}<br>Model B: %{x}<br>Win Rate: %{z}<extra></extra>")
    fig.write_image(args.output)

    




    runs = 1000
    ratings = None
    for _ in range(runs):
        sampled_pairwise_comps = random.choices(pairwise_comps, k=len(pairwise_comps))
        c_ratings = compute_elo(sampled_pairwise_comps, K=4, SCALE=400, BASE=10, INIT_RATING=1000)
        if ratings is None:
            ratings = {key: [c_ratings[key]] for key in c_ratings}
        else:
            ratings = {key: ratings[key] + [c_ratings[key]] for key in c_ratings}
        
    ratings = {key: np.median(ratings[key]) for key in ratings}



    sorted_ratings = dict(sorted(ratings.items(), key=lambda item: item[1], reverse=True))

    if args.reranking: #all_datasets = [] ## filename, data outputs.
        fname_content = {}
        for d in all_datasets:
            fname_content[d[0]] = d[1]
            
        top_model = max(sorted_ratings, key=sorted_ratings.get)
        refs = fname_content[top_model]
        reranked_models = {}
        for k, v in sorted_ratings.items():
            if k != top_model:
                
                s = score(fname_content[k], refs, lang='en', verbose=True, model_type=args.model_type, batch_size=16)[2].numpy()
                reranked_models[k] = np.mean(s)
                print(f"{k}: {np.std(s)}")
                
        reranked_models = dict(sorted(reranked_models.items(), key=lambda item: item[1], reverse=True))
        



    print()
    print(args)
    print(f'All compares: {len(pairwise_comps)}')
    print('===== Elo Rankings =====')
    print("{:>15}: Rating".format("Model"))
    for i, (k, v) in enumerate(sorted_ratings.items()):
        print(f'{i+1}. {k:>15}: {v:.1f}')
    
    

    sorted_wins = dict(sorted(num_wins.items(), key=lambda item: item[1], reverse=True))
    print('===== Num of wins =====')
    print("{:>15}: Wins".format("Model"))
    for i, (k, v) in enumerate(sorted_wins.items()):
        print(f'{i+1}. {k:>15}: {v}')

    if args.reranking:
        print()
        print(f'===== Rerankings by top model: {top_model} =====')
        print("{:>15}: Rating".format("Model"))
        for k, v in reranked_models.items():
            print(f'{k:>15}: {v:.6f}')

    
    print(f'Saved eval result to {args.output}')


if __name__ == '__main__':
    main()      
        

















