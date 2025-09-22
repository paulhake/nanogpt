# nanoGPT

This repository is based on the groundbreaking nanoGPT lecture series by Andrej Karpathy, where he demonstrates how to build a GPT-like language model from scratch.

## About Andrej Karpathy's nanoGPT Project

Andrej Karpathy, former Director of AI at Tesla and co-founder of OpenAI, created nanoGPT as an educational initiative to demystify how large language models work. His approach is uniquely hands-on: starting from an empty file and building up to a reproduction of GPT-2 (124M parameters) that can be trained in approximately 1 hour for around $10.

The project embodies Karpathy's philosophy of "spell it out" education - where complex concepts are broken down into digestible, implementable steps. Through his YouTube tutorial, he live-codes the entire process, explaining each component of the transformer architecture as it's implemented. This makes advanced AI concepts accessible to developers who want to understand the fundamentals rather than just use black-box APIs.

What makes nanoGPT special is its minimalism combined with completeness. Built in PyTorch with minimal dependencies, it centers on just a few key files while still implementing the full transformer architecture from the seminal "Attention is All You Need" paper. The codebase serves as both a learning tool and a practical starting point for experimentation with GPT models.

## Implementation Details

This repository contains a simple bigram language model implementation following Karpathy's educational tutorials. The code demonstrates fundamental concepts of transformer-based language models using PyTorch, trained on Shakespeare text to showcase the basic principles of autoregressive language modeling.

## Files

- `bigram.py` - Simple bigram language model implementation
- `bigramv2.py` - Enhanced version of the bigram model
- `gpt-dev.ipynb` - Development notebook for experimentation
- `input.txt` - Training data (Shakespeare text)
- `requirements.txt` - Python dependencies

## Usage

```bash
pip install -r requirements.txt
python bigram.py
```

## Key Resources and References

### Primary Sources
- **Original nanoGPT Repository**: [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) - The main implementation for training/finetuning medium-sized GPTs
- **Tutorial Repository**: [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt) - Video+code lecture building nanoGPT from scratch
- **YouTube Tutorial**: [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) - Complete walkthrough by Andrej Karpathy

### Foundational Papers
- **"Attention is All You Need"** - The original transformer architecture paper
- **"Language Models are Few-Shot Learners"** - GPT-3 paper demonstrating few-shot learning capabilities

### Community Resources
- **#nanoGPT Discord** - Active community for questions and discussions
- **Medium Tutorial**: [Train your own language model with nanoGPT](https://sophiamyang.medium.com/train-your-own-language-model-with-nanogpt-83d86f26705e) - Hands-on walkthrough by Sophia Yang
- **DoltHub Blog**: [Exploring NanoGPT](https://www.dolthub.com/blog/2023-02-20-exploring-nanogpt/) - Technical exploration inspired by Karpathy's tutorial

### Additional Learning Materials
- **Simon Willison's Tutorial**: [Training nanoGPT on custom content](https://til.simonwillison.net/llms/training-nanogpt-on-my-blog) - Practical application examples
- **Community Implementations**: [Various forks and educational versions](https://github.com/gs-101/nanoGPT-from-scratch) available on GitHub

## Educational Philosophy

This project reflects Karpathy's commitment to education and making AI accessible. As he often emphasizes, the goal is not just to use AI tools, but to understand how they work at a fundamental level. nanoGPT serves as a bridge between theoretical knowledge and practical implementation, designed for "people who want to experiment with building their own GPT models without needing massive hardware or deep expertise."