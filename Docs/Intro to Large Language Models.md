# Intro to Large Language Models

Created by: Nick Durbin
Created time: July 12, 2024 4:43 PM
Last edited by: Nick Durbin
Last edited time: July 25, 2024 1:59 AM
Tags: Intro

# LLM Inference

An LLM is two files:

- Parameters - large list of weights for the neural network
- run.c - file that executes neural network
    - Can be written in any language (c, python, etc.)

Can be run with only using two files and a device (no internet needed)

<aside>
💡 Computational complexity comes from retrieving the parameters. The run file is fairly simple.

</aside>

# LLM Training

- Training is the process of obtaining the right parameters
- Inference is running it on a device, training is a computationally very involved process

Training involves getting a chunk of data and then in a way compressing (lossy compression) it using a GPU cluster.

<aside>
💡 Llama 2 70B numbers from Meta AI: 6000 GPUs for 12 days, ~$2M, ~1e24 FLOPS. 
*Modern models take way longer*

</aside>

## Neural Network

- Tries to predict next word in sequence
- Next word prediction forces the neural network to learn a lot about the world
- If given a document the network will have to learn specific facts contained in the document in order to predict the next word correctly
- Facts get compressed into the weights
- The network “dreams” internet documents

# How do the neural networks work?

Little is known in full detail

- Billions of parameters are dispersed through the network
- We know how to iteratively adjust them to make it better at prediction
- We can measure that this works, but we don’t really know how the billions of parameters collaborate to do it

They build and maintain some kind of knowledge database, but it is a bit strange and imperfect:

Recent viral example: “reversal curse”

> Q: “Who is Tom Cruise’s mother”?
A: Mary Lee Pfeiffer
> 

> Q: “Who is Mary Lee Pfeiffer’s son?”
A: I don’t know
> 

**LLMs are mostly inscrutable artifacts, develop correspondingly sophisticated evaluations.**

# Finetuning into an Assistant

So far only internet document generators were discussed

Stages of training:

1. Pre-training - training an internet document generator, like the one discussed
2. Finetuning - obtaining an assistant model and make it answer questions

Training is the same as before (predicting next word), but with a swapped dataset that is collected manually by using lots of people.

Engineers at a company will come up with labeling documentations that ensure the proper format to finetune the model. In other words, high quality data is created by people based on labeling instructions.

Pre-training trains a model with a lot but low quality data, while finetuning trains the model on high quality but low quantity data.

 

Assistant model is trained on Q and A data during finetuning.

# Summary so far

**Stage 1: Pretraining (~every year since expensive)**

1. Download ~10TB of text
2. Get a cluster of ~6,000 GPUs
3. Compress the text into a neural network, pay ~$2M, wait 12 days
4. Obtain **base model**

**Stage 2: Finetuning (~every week)**

1. Write labeling instructions
2. Hire people (or use a GPT or scale.ai!), collect 100K high quality ideal Q&A responses, and/or comparisons
3. Finetune base model on this data, wait ~1 day
4. Obtain **assistant model**
5. Run a lot of evaluations
6. Deploy
7. Monitor, collect misbehaviors, go to step 1

*It is often much easier to compare Answers instead of writing Answers*

**Optional Stage 3: Comparisons** Reinforcement Learning from Human Feedback (RLHF)

Uses comparison labels to improve model based of already generated output

<aside>
💡 How to fix misbehaviors:
1. Take conversation where misbehavior happened
2. Ask person to fill in correct response
3. Overwritten response is inserted into training data
4. Model improves on next iterations

</aside>

# More LLM facts

**Increasingly, labeling is a human-machine collaboration…**

- LLMs can reference and follow the labeling instructions just as humans can
- LLMs can create drafts, for humans to slice together into a final label
- LLMs can review and critique labels based on their instructions

**LLMs leaderboards also exist where score (or elo) is calculated in a similar way to chess.**

- Two models answer the same question and a human picks which one is better

Currently proprietary models (closed) do better and the open models in a way chase them.

# LLM Scaling Laws

Performance of LLMs is a smooth, well-behaved, predictable function of:

- N - the number of parameters in the network
- D - the amount of text we train on

The trends do not show signs of “topping out”

**⇒ We can expect more intelligence “for free” by scaling**

# Tool use

- Based on different requests, an LLM might used specific tools to make it more capable
    - It might use a browser to collect info
    - It might use a calculator to solve math equations
    - It could write code in Python with a specific library to make a plot

Similar to a human that uses tools to simplify tasks, an LLM can use existing computing infrastructure for problem solving

# Multimodality (Vision, Audio)

- Current LLMs have the ability to write functional HTML code based of a sketch of a website on paper
- Speech to speech communication is also another way to interface with a model

# Thinking, System 1/2

- System 1: fast, instinctive thinking without analyzing a tree of possibilities
- System 2: slower, but more accurate thinking that tries to analyze different scenarios

Currently LLMs only operate on a System 1 level. Many people are inspired by the idea that an LLM will be able to think through a tree of possibilities which will cause it to provide slower, but more accurate results.

# Self-improvement

**AlphaGo had two major stages:**

1. Learn by imitating expert human players
2. Learn by self-improvement (reward = win the game)

**Big question in LLMs:**

What does Step 2 look like in the open domain of language?

Main challenge: Lack of reward criterion (reward function)

# LLM customization

Currently one might create a custom GPT by giving prompts and data for the model to browse and use in its responses.

In the future people might be able to fully train these models based on their needs

# LLM OS

It is not accurate to think of an LLM as a chatbot or word generator

An LLM is a kernel process of an emerging operating system. This process is coordinating a lot of resources (memory or computational tools) for problem solving.

**An LLM in a few years:**

It can read and generate text

It has more knowledge than any single human about all subjects

It can browse the internet

It can use the existing software infrastructure (calculator, Python, mouse/keyboard)

It can see and generate images and video

It can hear and speak, and generate music

It can think for a long time using a System 2

It can “self-improve” in domains that offer reward function

It can be customized and finetuned for specific tasks, many versions exist in the app stores

It can communicate with other LLMs

# Jailbreak

Is deceiving a model into making it give out information that it is not meant to give out.

Models are fluent in base 64 and some might get jailbreaked if this “language” is used.

For example Claude’s refusal data was mostly in English and therefore it couldn’t refuse a harmful query in another language

Some requests would not refuse harmful queries after a Universal Transferable Suffix was added that jailbreaks any request

These jailbreaks can also be achieved by adding an image of a panda to the request with a specific noise pattern

You can imagine reoptimising and getting a different nonsense pattern to jailbreak the models

# Prompt Injection

Is hijacking a model by giving it what looks like mute instructions and taking over a prompt

**Attack examples:**

Some websites might contain hidden prompt injections to change an LLM response during browsing to offer fraud links

Bard will exfiltrate data through a google doc that has a prompt injection

# Data poisining

“Sleeper agent” attack

1. An attacker might hide a carefully crafted text with a custom trigger phrase, e.g. “James Bond”
2. When this trigger word is encountered at test time, the model outputs become random, or changed in a specific way

These attacks are kind of a cat and mouse game that can be seen in traditional online security and now are becoming relevant in LLM security.