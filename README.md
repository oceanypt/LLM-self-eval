# A Self-Evaluation Method for Large Language Models

By [**Hai Ye**](https://oceanypt.github.io/), from National University of Singapore.

We present an automatic evaluation method for evaluating instruction-following ability of large language models. 

Different from using GPT-4 as the evaluator, our method utilizes the model themselves to do self evaluation. 



## Introduction

Currently, there are more and more open-sourced and not open-sourced instruction-following models, such as ChatGPT, GPT-4, Claude, Bard, Vicuna, Alpaca, etc. These models are fine-tuned on pre-trained large language models such as GPT-3 and LLaMA, with instructions labeled by humans or distilled from larger models (e.g., ChatGPT). 

How to evaluate these models becomes an issue and attracts more attention recently. Evaluating the outputs from these instruction-following models is not easy, since the outputs of the models are open-ended and actually there are no gold answers. 

Recently, a benchmark called **Arena** tries to bridge the gap by using human evaluation in the wild. Each time a user asks a question, the system out of the benchmark will sample two models to generate two separate outputs for the question. Then the user has to make a preference over the two model outputs. The preferred model wins the not preferred model. By collecting many pairs of these comparisons, the Elo ratings will be calculated using these results of pairwise comparisons. 

One obvious **drawback** of this kind of evaluation methodology lies in that it is quite costly to conduct human evaluations and the whole process is not reproducible. 



## Auto-evaluation by Model Themselves

Here, in this work, we propose a new insight that tries to let models evaluate themselves. Similar to Arena, we also conduct pairwise comparisons among models, but the difference is that we don’t require humans to leave preference feedback anymore, but let the rest of the models that don’t participate in the comparison do evaluation. 


## Method

**Notation**. Suppose we have $N$ models for evaluation, and model $i$ is denoted as $f_i$. We also have an evaluation set $D=\\{x_t\\}_{t=1}^M$. We also have an evaluation metric $s(o_i,o_j)$ to compare two model outputs which are $o_i$ and $o_j$.

**Algorithm**. So, every time, for the input $x_t$, we sample two models which are $f_i$ and $f_j$, and we obtain the outputs from the two models respectively, $o_i(x_t)$ and $o_j(x_t)$. Then we also randomly sample a model $f_k$ from the rest $N-2$ models, and also obtain its output $o_k(x_t)$. 

For the input $x_t$, with pairwise models $f_i$ and $f_j$, we utilize model $f_k$ to decide which model will win by the following:

- $f_i \rightarrow \text{win}, \text{if} \ s(o_i, o_k) > s(o_j, o_k);$
- $f_j \rightarrow \text{win}, \text{if} \ s(o_i, o_k) < s(o_j, o_k);$
- $\text{tie}, \text{if} \ s(o_i, o_k) = s(o_j, o_k).$

where $s(\cdot, \cdot)$ is the BERT-score, which here is to measure the token overlap between two outputs. We regard model $k$ prefers model $i$ as its output has a higher overlap with the output of model $i$; otherwise, model $k$ prefers model $j$. Here, if $o_k$ is seen as the reference, then we use the recall score from BERT-score, and if $o_k$ is seen as the hypothesis, then we use the precision score. In this way, we only focus on the token overlap between two model outputs, without considering the output length of model $i$ and model $j$.

So, repeating all evaluation data ($M$), all pairwise model samples ($C_N^2$), and all rest of the models as the judge ($N-2$), we can obtain total $M * C_N^2 * (N-2)$ pairs, each records two model names and the comparison result. 

Finally, with these compare results, we also calculate the Elo ratings to obtain the final model rankings.

## Improvement

The weak models may not be capable of evaluating strong models. In theory, weak models can evaluate the strong models, but they may introduce some noise. **If a model is very weak compared to two evaluated models, then it may not be so accurate for the results.** **But if the performance of models is not so different, then the evaluation would be more robust.** 

The solution can be:

- We first do an evaluation without saying which model is strong or which model is weak. We obtain the first-round rankings and results.
- Then, we only keep the top models and do the second-round evaluation.



## Code Usage
To use our code, you should prepare the outputs of different models under the same folder. Each file corresponds to one model, and each line in a file corresponds to the model output of one input instruction. After that, run as follows:
```bash
python self_evaluation.py -i ./responses  -b 0. -m bert-score -f allenai/longformer-large-4096
```
Here, we use BERT-score as the evaluation metric, you can also use Rouge score. 

By running the above code, you would see the **Elo rating** of the tested models.


To know more information of the arguments, run:
```bash
python self_evaluation.py -h
```




## Citation
```
@misc{HaiYe-LLM-self-eval,
  author = {Hai Ye},
  title = {A Self-Evaluation Method for Large Language Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/oceanypt/LLM-self-eval}}
}
```

