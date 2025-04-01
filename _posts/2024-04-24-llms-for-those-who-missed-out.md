---
layout: post
title: LLMs for Those Who Missed Out
date: 2024-04-24 15:09:00
description: Let's talk about large language models. Once again.
tags: nlp, llm
categories: old-posts
featured: false
---

An LLM (large language model) consists of just two files:

- Large file with billions of parameters (weights)
- Small file with about 500 lines of code to run the model

LLM doesn’t take up much space and doesn’t require extensive computing power.

For example, the [Llama-2–70b model](https://huggingface.co/meta-llama/Llama-2-70b-chat-hf) from Meta, which is open source and has 70 billion parameters, is just 140 GB and can be run locally on a MacBook without internet access. You can [download this model](https://www.llama.com/llama-downloads/) from Meta’s website and use it for free.

However, you need significant computing resources to get these parameters (train the model).

## How to Train a Model

### Stage 1: Pretraining (Training the base model)

Think of this process as compressing the Internet into a neural network, similar to how a ZIP file compresses documents into an archive.

Here’s what you need to get a model like Llama-2:

- Take a “slice of the internet,” approximately ten terabytes of text in size
- Use a cluster of ~6,000 GPUs (specialized graphics processors used for heavy computations)
- Compress the text into the neural network. This takes about 12 days and costs around $2M
- Acquire the base model (the file with parameters)

The primary function of the base model is to predict the next word. You input a sequence of words, and it outputs the next most likely word based on the texts it was trained on.

For example, for the phrase “cat sat on a,” the LLM might predict the word “mat” with 97% probability.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-04-24-llms/0.png" title="Image from the presentation illustrating how the neural network works" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Image from the presentation illustrating how the neural network works
</div>

That’s basically how compression works: if you can predict the next word accurately, you can use this information to compress the original data.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-04-24-llms/1.png" title="I checked, and ChatGPT indeed responds with “mat”" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    I checked, and ChatGPT indeed responds with “mat”
</div>


However, unlike a ZIP file, where compression is lossless, LLM “compresses” the internet with losses. This is why models can hallucinate: they make up information that doesn’t actually exist but appears plausible (similar to what they have seen in the training data).

Moreover, models can sometimes produce completely unexpected things. For example, ChatGPT knows the answer to the question, “Who is Tom Cruise’s mother?” (Mary Lee Pfeiffer), but it doesn’t know the answer to the question, “Who is Mary Lee Pfeiffer’s son?”

This isn’t a typical database that simply stores information; it’s a different format we don’t fully understand. 

This clearly illustrates that we don’t quite grasp how this whole thing works and can only see the results it produces.

### Stage 2: Finetuning (Additional training)

The base model isn’t very practical for everyday use. We don’t just want to receive continuous word sequences; we want to ask questions and get answers.

This requires finetuning — a process in which we develop an assistant model that answers questions.

The training process for the assistant model is similar to that of the base model, but now we train the model not on internet texts but on data we manually collect. Companies hire people who write questions and answer them.

If the training of the base model occurs on huge volumes of often low-quality text, the training of the assistant model involves comparatively smaller volumes (say, 100,000 documents), but they are all of very high quality.

After such finetuning, the model understands that when asked a question, it should respond in the style of a helpful assistant.

Unlike the training of the base model, which is conducted at most 2–3 times a year, finetuning can be done regularly, even weekly, as it is much less costly.

### Stage 3 (optional): Comparisons
In many cases it’s easier not to write a response from scratch but to compare several different answer options and choose the best one.

The data from these comparisons is used to train the model further.

At OpenAI, this is called RLHF (Reinforcement Learning from Human Feedback).

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-04-24-llms/2.png" title="Difference between Open and Closed Models" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Difference between Open and Closed Models. Source: Chatbot leaderboards
</div>


Models with open weights, such as Llama-2, are still inferior compared to proprietary models like GPT-4 and Claude.

However, proprietary models cannot be downloaded for personal use or further training; they can only be operated through a web interface or an API (though the introduction of Custom GPTs at OpenAI is a first step towards customization)

## Capabilities of LLMs (using ChatGPT as an example)

Prompted by an LLM, it can understand what the user needs and use external tools for this:

- Search the internet if the user needs up-to-date information (goes to Bing, retrieves a list of the latest links for the query, copies the full text of each link into its context window, and summarizes the information)
- Use a calculator to do calculations
- Use a Python library to draw graphs
- Draw images using DALL-E
- Write code

Moreover, LLMs can be multimodal. They can recognize text, voice, images, or video and respond with voice, images, or video.

## Can LLMs Make Decisions?

There’s a well-known book by Kahneman, “Thinking, Fast and Slow.”

The main idea of the book is that there are two systems in the brain: System 1, which has fast, automatic reactions, and System 2, which is slower, rational, and conscious and makes complex decisions.

For example, the answer to question 2+2 comes from System 1 because this knowledge is automatic for us. But calculating 17×24 requires System 2.

If we apply these terms, current LLMs only possess System 1. They can only provide the most likely next word in real time.

It would be great if we could come to an LLM and say: here’s my question; you can think for 30 minutes, but I need a very accurate and high-quality answer.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/posts/2024-04-24-llms/3.png" title="Thought tree" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Thought tree
</div>

No models can do that yet. But it would be desirable for a model to have something like a “thought tree,” through which it could navigate, analyze the results, go back and try again until it achieves a result it is most confident about.

## Can LLMs Train Themselves?
There’s a famous case where AlphaGo (a program that plays Go) was trained in two stages:

First, it was trained on human games and learned to play very well. Then it began to train itself-playing against itself, trying to maximize the likelihood of winning-and significantly improved its quality.

But in LLMs, we’re only at stage 1-training only occurs on materials created by humans. 

Why is this?

In the game of Go, there is a very clear criterion for success-a won game- and you can train the model to maximize the likelihood of winning. 

In LLMs, the criteria are not so obvious. It’s not clear how to assess whether the result is good.

Such criteria can be found in some narrow topics, but in general, it’s still hard to imagine.

## What LLMs Will Be Able to Do in a Few Years
Finally, a small forecast.

In a few years, LLMs:

- Will have more knowledge than any human on all subjects
- Can operate on the internet
- Can use existing software infrastructure (calculator, Python, mouse/keyboard)
- Can see and generate images and videos
- Can hear and speak, and also generate music
- Can think for a long period using System 2
- Can train themselves in areas where there are clear criteria
- Can be customized and refined for specific tasks. Many versions will exist in app stores
- Can interact with other LLMs