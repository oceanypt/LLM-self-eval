# A Self-Evaluation Method for Large Language Models

We present an automatic evaluation method for evaluating instruction-following ability of large language models. 

Different from using GPT-4 as the evaluator, our method utilizes the model themselves to do self evaluation. 




## Method

Notation. Suppose we have $N$ models for evaluation, and model $i$ is denoted as $f_i$. We also have an evaluation set $D=\{x_t\}_{t=1}^M$. We also have an evaluation metric $s(o_i,o_j)$ to compare two model outputs which are $o_i$ and $o_j$.

**Algorithm**. So, every time, for the input $x_t$, we sample two models which are $f_i$ and $f_j$, and we obtain the outputs from the two models respectively, $o_i(x_t)$ and $o_j(x_t)$. Then we also randomly sample a model $f_k$ from the rest $N-2$ models, and also obtain its output $o_k(x_t)$. 

For the input $x_t$, with pairwise models $f_i$ and $f_j$, we utilize model $f_k$ to decide which model will win by the following:

$f_i \rightarrow \text{win}, \text{if} \ s(o_i, o_k) > s(o_j, o_k); \\ f_j \rightarrow \text{win}, \text{if} \ s(o_i, o_k) < s(o_j, o_k); \\ \text{tie}, \text{if} \ s(o_i, o_k) = s(o_j, o_k).$

where $s(\cdot, \cdot)$ is the BERT-score, which here is to measure the token overlap between two outputs. We regard model $k$ prefers model $i$ as its output has a higher overlap with the output of model $i$; otherwise, model $k$ prefers model $j$. Here, if $o_k$ is seen as the reference, then we use the recall score from BERT-score, and if $o_k$ is seen as the hypothesis, then we use the precision score. In this way, we only focus on the token overlap between two model outputs, without considering the output length of model $i$ and model $j$.

So, repeating all evaluation data ($M$), all pairwise model samples ($C_N^2$), and all rest of the models as the judge ($N-2$), we can obtain total $M * C_N^2 * (N-2)$ pairs, each records two model names and the comparison result. 

Finally, with these compare results, we also calculate the Elo ratings to obtain the final model rankings.

