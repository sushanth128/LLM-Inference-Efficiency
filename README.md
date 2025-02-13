# LLM Efficiency Enhancement: Speculative Decoding Model Generation

<img width="358" alt="image" src="https://github.com/user-attachments/assets/ed8ab786-c8b4-47b4-9ad5-b60623a18409" />

# Table of Contents
1. [Overview](#overview)
2. [Executive Summary](#executive-summary)
3. [Project Goals](#project-goals)
4. [Approach](#approach)
5. [Results](#results)
6. [Conclusion](#conclusion)
7. [Key Technologies](#key-technologies)
8. [Lessons Learned](#lessons-learned)

## Overview

- Project Title: LLM Efficiency Enhancement: Speculative Decoding Model Generation
- Authors: Arian Raje, Sushanth Gangireddy, Victor Akinwande
- Date: August 25, 2024

## Executive Summary

Autoregressive decoding in Large Language Models (LLMs) is inherently slow due to the sequential generation of tokens, where only one token is generated at a time. This creates significant delays when generating long sequences. Speculative decoding offers an alternative approach that uses a smaller, more efficient draft model to generate tokens quickly. The target model, the original LLM, then verifies the generated tokens in parallel, accepting or rejecting them.

This project addresses a gap in speculative decoding research, focusing on constructing an effective draft model for arbitrary target models, rather than relying on pre-existing draft models with identical vocabularies and training data. We explored traditional model compression techniques, such as pruning, quantization, and layer compression, and evaluated their performance within the speculative decoding framework. Our findings indicate that selecting a draft model that strikes an optimal balance between model size and utility leads to consistent improvements in inference speed.

## Project Goals

- Develop a Draft Model: Construct a smaller, more efficient draft model to assist with speculative decoding for an arbitrary target LLM.

- Optimize Model Size and Efficiency: Utilize traditional model compression techniques, such as pruning and quantization, to create an efficient draft model.

- Evaluate Speedups: Analyze the impact of speculative decoding on inference speed compared to traditional autoregressive decoding methods.

## Approach

To tackle the problem of slow autoregressive decoding, we used speculative decoding, which decouples token generation from the target model, allowing faster token production by leveraging a draft model. The draft model produces tokens, which are then verified and refined by the target model.

We focused on creating a draft model using model compression techniques to reduce its size and complexity. The methods we employed include:

- Pruning: Removing unimportant parameters or neurons in the model.

<img width="474" alt="image" src="https://github.com/user-attachments/assets/321c7e9a-f861-4a70-8211-74abb58e788f" />

- Quantization: Reducing the precision of model weights to make the model more lightweight.

<img width="550" alt="image" src="https://github.com/user-attachments/assets/eb88e191-377b-420e-b799-81f166937756" />

- Layer Compression: Compressing or reducing the depth of neural network layers.

<img width="542" alt="image" src="https://github.com/user-attachments/assets/5101a1e7-1929-4e5c-9142-a7aa978fca67" />

These techniques were applied to ensure that the draft model could efficiently generate tokens, while the target LLM verified them for accuracy. Our evaluation focused on determining the trade-offs between the draft model’s size and its impact on the performance of the target model, measuring inference speedups.

## Results

<img width="318" alt="image" src="https://github.com/user-attachments/assets/d90088f5-bb5c-42c0-a10b-d46dbea78b71" /> <img width="318" alt="image" src="https://github.com/user-attachments/assets/5eb1d1bd-a765-4719-bf51-84e61ec1cd7f" />

<img width="464" alt="image" src="https://github.com/user-attachments/assets/35e19910-60c3-4ad0-8876-1e75a959626d" />


Our experiments demonstrated that speculative decoding using a compressed draft model resulted in significant speedups in inference, with minimal degradation in output quality. By carefully selecting the compression technique that best balanced the draft model's size and utility, we achieved consistent improvements in performance. This approach allows for faster inference without compromising the capabilities of the original target model.

## Conclusion

This project explores the use of speculative decoding for enhancing the efficiency of LLM inference. By creating an effective draft model using model compression techniques, we were able to accelerate the decoding process, making it more efficient for large-scale applications. Future work may include exploring advanced compression methods and further optimizing the draft model’s performance across different target models.

## Key Technologies

PyTorch: Framework for model development and training.
Model Compression Techniques: Pruning, quantization, and layer compression.
Speculative Decoding: Framework for enhancing inference speed in LLMs.
Data Usage
The project relies on pre-trained language models, such as LLaMaA2 and GPT-2, to evaluate the impact of compression techniques. These models were fine-tuned and tested under the speculative decoding framework to benchmark the effectiveness of various compression methods. Evaluation metrics include speed improvements in inference time and semantic performance measured by accuracy in token generation.

## Lessons Learned

Model compression techniques can significantly speed up the inference process, but selecting the appropriate balance between compression and model utility is critical.
Speculative decoding presents a promising path toward optimizing LLM inference efficiency, particularly for large models.

## Future Work
Investigate more advanced compression techniques and hybrid methods to further improve draft model efficiency.
Extend the framework to work with a broader set of LLMs and target models, potentially integrating with deployment environments for real-time applications.
