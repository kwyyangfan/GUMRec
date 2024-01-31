# LLMRec

## Abstract

Personalized recommendation plays a crucial role in Internet platforms, providing users with tailored content based on their user models and enhancing user satisfaction and experience. To address the challenge of information overload, it is essential to analyze user needs comprehensively, considering not only historical behavior and interests but also the user's intentions and profiles. Previous user modeling approaches for personalized recommendations have exhibited certain limitations, relying primarily on historical behavior data to infer user preferences, which results in challenges such as the cold start problem, incomplete modeling, and limited explanation. Motivated by recent advancements in large language models (LLMs), we present a novel approach to user modeling by embracing generative user modeling using LLMs. We propose Generative User Modeling with Chain-of-Thought Prompting for Personalized Recommendation, which utilizes LLMs to generate comprehensive and accurate user models expressed in natural language and then employs these user models to empower LLMs for personalized recommendation. Specifically, we adopt the chain-of-thought prompting method to reason about user attributes, subjective preferences, and intentions, integrating them into a holistic user model. Subsequently, we utilize the generated user models as input to LLMs and design a collection of prompts to align the LLMs with various recommendation tasks, encompassing rating prediction, sequential recommendation, direct recommendation, and explanation generation. Extensive experiments conducted on real-world datasets demonstrate the immense potential of large language models in generating natural language user models, and the adoption of generative user modeling significantly enhances the performance of LLMs across the four recommendation tasks. 

<img src="https://github.com/kwyyangfan/GUMRec/master/docs/Overall Framework-改.png" width="860" />

## Generative User Modeling with Chain-of-Thought

We expect the LLM to infer and integrate the three different aspects of user modeling. To achieve this, we employ chain-of-thought prompting in four logical steps as follows, reasoning about the user's attributes, preferences, and intentions.

<img src="https://github.com/kwyyangfan/GUMRec/master/docs/Generative User Modeling-改.png" width="860" />

## Prompt Construction for Recommendation Tasks

We design a collection of prompts to align the LLMs with various recommendation tasks, encompassing rating prediction, sequential recommendation, direct recommendation, and explanation generation.

<img src="https://github.com/kwyyangfan/GUMRec/master/docs/Prompt examples for four tasks-改.png" width="860" />

## Datasets Acquire

BaiduCloud: <https://pan.baidu.com/s/14ndz_wFpoHU0o5hCYp45lQ?pwd=qbt2> code: qbt2
