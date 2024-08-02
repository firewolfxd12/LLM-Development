# Let’s build the GPT tokenizer

Created by: Nick Durbin
Created time: July 30, 2024 1:11 PM
Last edited by: Nick Durbin
Last edited time: July 30, 2024 6:04 PM
Tags: GPT

## Byte-pair encoding

Byte-pair encoding is a tokenization algorithm used by GPT’s and Llama models. It is explained in the GPT2 paper and Llama2 paper. Tokens are the fundamental atom of LLM’s. 

Tokenization is at the heart of much weirdness of LLMs. Do not brush it off.

- Why can't LLM spell words? **Tokenization**.
- Why can't LLM do super simple string processing tasks like reversing a string? **Tokenization**.
- Why is LLM worse at non-English languages (e.g. Japanese)? **Tokenization**.
- Why is LLM bad at simple arithmetic? **Tokenization**.
- Why did GPT-2 have more than necessary trouble coding in Python? **Tokenization**.
- Why did my LLM abruptly halt when it sees the string "<|endoftext|>"? **Tokenization**.
- What is this weird warning I get about a "trailing whitespace"? **Tokenization**.
- Why the LLM break if I ask it about "SolidGoldMagikarp"? **Tokenization**.
- Why should I prefer to use YAML over JSON with LLMs? **Tokenization**.
- Why is LLM not actually end-to-end language modeling? **Tokenization**.
- What is the real root of suffering? **Tokenization**.

## Tokenization example

![Untitled](Let%E2%80%99s%20build%20the%20GPT%20tokenizer/Untitled.png)

In this example we see that the tokens are at the sub word level. They are completely arbitrary and might divide numbers and words in half, which might by the root of many problems.

Even if there are two words like “egg” that are exactly alike in the prompt, the tokenizer might assign the same words different tokens or divide it like before.

Also languages that are not in english work worse since the training set used during training was mostly in english, and the tokenization itself. Non english text will have tokens that are smaller by length also.

Coding languages like Python had problems with tokenization and it was due to the fact that tokenization was giving out too many tokens to unimportant characters like spaces. This made GPT2 bad at coding.

The tokenizer that GPT4 uses took this problem into account and now large spaces are encoded as a single character.

## Strings in Python, unicode code points

Strings in python are a sequence of unicode code points. Unicode code points are integers that are assigned to characters according to the Unicode standard.

So why can’t we just use this encoding for the LLM? One reason is that the vocabulary would be quite long with 149 813 characters. Furthermore, the standard is really alive and keeps changing. Its not a stable representation that we want to use.

The Unicode Standard defines three encodings: UTF-8, UTF-16, UTF-32. UTF-8 takes every single code point and translates it to a byte stream from 1-4 bytes. UTF-8 is preferred and is used more prominently on the internet. Other encodings are more wasteful for simple ASCII characters.

More on UTF-8:

[A Programmer’s Introduction to Unicode – Nathan Reed’s coding blog](https://www.reedbeta.com/blog/programmers-intro-to-unicode/)

So UTF-8 is the best choice, but we can’t use it naively. It only has 256 possible encodings for all the characters. We want to support a larger vocab size and use UTF-8. Hence the Byte-pair encoding algorithm. Some recent papers have proposed feeding raw bytes into transformers, without tokenization, but it hasn’t been proven yet. 

## Byte Pair Encoding (BPE) algorithm walkthrough

The goal is to compress the data. Lets say you pass this string to the tokenizer.

> aaabdaaabac
> 

The byte pair “aa” occurs most often, so we will replace it with a byte that does not exist in the data such as “Z”. This is called minting a new token.

> ZabdZabac
Z=aa
> 

Then the process is repeated with the byte pair “ab”, replacing it with “Y”.

> ZYdZYac
Y=ab
Z=aa
> 

The only literal byte pair left occurs only once, and the encoding might stop here. Alternatively, the process could continue with [recursive](https://en.wikipedia.org/wiki/Recursion) byte pair encoding, replacing "ZY" with "X":

> XdXac
X=ZY
Y=ab
Z=aa
> 

This data cannot be compressed further by byte pair encoding because there are no pairs of bytes that occur more than once.

To decompress the data, simply perform the replacements in the reverse order. 

Full code: https://colab.research.google.com/drive/1y0KnCFZvGVf_odSfcNAws6kcDD7HsI0L?usp=sharing#scrollTo=tzf3pOJmOhvo

## Tokenizer/LLM diagram

![Untitled](Let%E2%80%99s%20build%20the%20GPT%20tokenizer/Untitled%201.png)

<aside>
💡 Note, the Tokenizer is a completely separate, independent module from the LLM. It has its own training dataset of text (which could be different from that of the LLM), on which you train the vocabulary using the Byte Pair Encoding (BPE) algorithm. It then translates back and forth between raw text and sequences of tokens. The LLM later only ever sees the tokens and never directly deals with any text.

</aside>

You can also train the LLM on only encoded data from the Tokenizer and throw away the main dataset.

The tokenizer data should be representative of the data that is fed to the LLM.

## Regex patterns to force splits using regex patterns (GPT series)

```python
import regex as re
gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

print(re.findall(gpt2pat, "Hello've world123 how's are you!!!?"))
```

OpenAI used specific regex rules to split certain patterns. The regex separates common apostrophes, letters, numbers, long spaces for the tokenizer to process.

This is the result:

```python
['Hello', "'ve", ' world', '123', ' how', "'s", ' are', ' you', '!!!?'] 
```

It is unclear how OpenAI made sure the tokenizer worked, but it wasn’t as simple as chunking it up and BPing it.

## Tiktoken library from OpenAI

This library contains the official tokenizer for Chat GPT4. They did change the regex to be case insensitive and numbers are split up into a max of 3 characters to prevent long number sequences. Also the vocabulary size increased from 50k to 100k (roughly).

```python
"pat_str": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""",
```

- `?i:` is for case insensitivity
- `\p{N}{1,3}` max length of number is 3

## GPT-2 encoder.py released by OpenAI walkthrough

The encoder.py file also has a `vocab` of bytes just like our code, but is called `encoder` . And their `vocab.bpe` are our `merges`. Algorithmically it is very familiar. 

Unfortunately the code is messy, but the BPE tokenizer works exactly the same.

## Special tokens

In the vocabulary there is a special token called `<|endoftext|>`  . It is used as a delimiter of files, so that the LLM understands when texts end so that context is maintained. 

New custom tokens can be created in the tiktoken library the following way:

```python
cl100k_base = tiktoken.get_encoding("cl100k_base")

# In production, load the arguments directly instead of accessing private attributes
# See openai_public.py for examples of arguments for specific encodings
enc = tiktoken.Encoding(
    # If you're changing the set of special tokens, make sure to use a different name
    # It should be clear from the name what behaviour to expect.
    name="cl100k_im",
    pat_str=cl100k_base._pat_str,
    mergeable_ranks=cl100k_base._mergeable_ranks,
    special_tokens={
        **cl100k_base._special_tokens,
        "<|im_start|>": 100264,
        "<|im_end|>": 100265,
    }
)
```

Adding special tokens does involve model surgery so that it understands the new vocabulary.

## Sentencepiece library intro, used to train Llama 2 vocabulary

Commonly used because (unlike tiktoken) it can efficiently both train and inference BPE tokenizers. It is used in both Llama and Mistral series.

[sentencepiece on Github link](https://github.com/google/sentencepiece).

**The big difference**: sentencepiece runs BPE on the Unicode code points directly! It then has an option `character_coverage` for what to do with very very rare codepoints that appear very few times, and it either maps them onto an UNK token, or if `byte_fallback` is turned on, it encodes them with utf-8 and then encodes the raw bytes instead.

TLDR:

- tiktoken encodes to utf-8 and then BPEs bytes
- sentencepiece BPEs the code points and optionally falls back to utf-8 bytes for rare code points (rarity is determined by character_coverage hyperparameter), which then get translated to byte tokens.

(Personally I think the tiktoken way is a lot cleaner...)

## How to set vocabulary set? revisiting gpt.py transformer

The vocabulary size can’t be infinite since the transformer is trying to predict the probability distributions for every character. That would affect the model size. Also if the vocabulary size is large enough, each one of the tokens will become more and more rare. This might under train the model. There will be fewer and fewer examples for each individual token which will cause some vectors to be undertrained. 

What if we have a pretrained model and we want to extend the vocabulary size for finetuning? Special tokens might be added for tools and to preserve data in the model. It is common to freeze model weights and then train it on parts of the LLM to preserve the model architecture.

## Training new tokens, example of prompt compression

Sometimes feeding very long prompts to models can be inefficient. This problem can be solved with gist tokens. What you do is train the model by distilation. You introduce the new tokens and you train the model such that the behavior of the LLM is identical to the model with the long prompt. It is a compression technique using gist tokens. 

You don’t change the model you only use token embeddings to trigger the same long prompt by only using some tokens. 

## Multimodal [image, video, audio] tokenization with vector quantization

To make models multimodal, most people are starting to converge to the fact that you can just convert images or audio to tokens and then train the same transformer like you would before but with text. 

The Sora model from OpenAI utilizes this technique only with videos. It chunkates the video then feeds it into a transformer.

## Revisiting and explaining the quirks of LLM tokenization

If you feed data in a specific way, it might break the transformer. For example inputs with trailing spaces cause bad model performance. This is because the model splits up the words to have spaces before words. 

### Examples:

“Oh “ will have worse performance than “Oh”. The transformer rarely saw information with trailing spaces which alters its behavior.

Special tokens like `.DefaultCellStyle` will cause the model to have bad performance. This is due to the fact that that word was really rear in the training data.

In some repositories there is a lot of error handling for special characters to prevent bugs.

There are also a lot of Trigger words that significantly affect the models behavior, sometimes causing it to swear at you or tell strange words like SolidGoldMagikarp

## **Final recommendations**

- Don't brush off tokenization. A lot of footguns and sharp edges here. Security issues. Safety issues.
- Eternal glory to anyone who can delete tokenization as a required step in LLMs.
- In your own application:
    - Maybe you can just re-use the GPT-4 tokens and tiktoken?
    - If you're training a vocab, ok to use BPE with sentencepiece. Careful with the million settings.
    - Switch to minbpe once it is as efficient as sentencepiece :)