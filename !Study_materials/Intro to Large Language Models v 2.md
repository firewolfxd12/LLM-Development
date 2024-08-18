# Intro to Large Language Models v.2

Created by: Jovan Plavsic
Created time: July 13, 2024 12:44 AM
Last edited by: Nick Durbin
Last edited time: July 25, 2024 1:59 AM
Tags: Intro

[https://github.com/karpathy/ng-video-lecture](https://github.com/karpathy/ng-video-lecture)

[Research Paper - Attention Is All You Need](https://arxiv.org/pdf/1706.03762) 

[YouTube Video- Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)  

**References**

![Untitled](Intro%20to%20Large%20Language%20Models%20v%2/03de747e-1787-41df-bc2e-61c84ed9399d.png)

![Untitled](Intro%20to%20Large%20Language%20Models%20v%2/26b12475-2aa0-46f1-858d-c41c04c12fb1.png)

**Large Scale Training**

---

input is ([18]) the target: 47
input is ([18, 47]) the target: 56
input is ([18, 47, 56]) the target: 57

block [18, 47, 56, 57]

```python
x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]
    print(f"when input is {context} the target: {target}")
```

1. Grab chunks of encoded text and find what character follows after every sequential (and starting from first character) combination of characters 
2. Encode all data into list of integers. 
    
    You can convert them in many different ways, for example
    
    - By character - a: 0, b: 1, c: 2….
    - By subword - ‘text’: 9, ‘ization’: 30…
    
    Methods like SentencePiece, tokenizer, tensorflow subword encoder
    
3. Find sufficiently large data set
    
    [cc-citations/bib at main · commoncrawl/cc-citations](https://github.com/commoncrawl/cc-citations/tree/main/bib)
    

![Untitled](Intro%20to%20Large%20Language%20Models%20v%2/Untitled.png)

Figure 1: The Transformer - model architecture

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

GPT — Generatively Pre-trained Transformer