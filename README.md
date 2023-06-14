# A Self-Evaluation Method for Large Language Models

We present an automatic evaluation method for evaluating instruction-following ability of large language models. 

Different from using GPT-4 as the evaluator, our method utilizes the model themselves to do self evaluation. 




## Method

Notation. Suppose we have $N$ models for evaluation, and model $i$ is denoted as $f_i$. We also have an evaluation set $D=\{x_t\}_{t=1}^M$. We also have an evaluation metric $s(o_i,o_j)$ to compare two model outputs which are $o_i$ and $o_j$. 