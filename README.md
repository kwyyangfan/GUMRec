# LLMRec

## Abstract

Personalized recommendation plays a crucial role in Internet platforms, providing users with tailored content based on their user models and enhancing user satisfaction and experience. To address the challenge of information overload, it is essential to analyze user needs comprehensively, considering not only historical behavior and interests but also the user's intentions and profiles. Previous user modeling approaches for personalized recommendations have exhibited certain limitations, relying primarily on historical behavior data to infer user preferences, which results in challenges such as the cold start problem, incomplete modeling, and limited explanation. Motivated by recent advancements in large language models (LLMs), we present a novel approach to user modeling by embracing generative user modeling using LLMs. We propose Generative User Modeling with Chain-of-Thought Prompting for Personalized Recommendation, which utilizes LLMs to generate comprehensive and accurate user models expressed in natural language and then employs these user models to empower LLMs for personalized recommendation. Specifically, we adopt the chain-of-thought prompting method to reason about user attributes, subjective preferences, and intentions, integrating them into a holistic user model. Subsequently, we utilize the generated user models as input to LLMs and design a collection of prompts to align the LLMs with various recommendation tasks, encompassing rating prediction, sequential recommendation, direct recommendation, and explanation generation. Extensive experiments conducted on real-world datasets demonstrate the immense potential of large language models in generating natural language user models, and the adoption of generative user modeling significantly enhances the performance of LLMs across the four recommendation tasks. 

<img src="https://github.com/kwyyangfan/GUMRec/blob/master/docs/Overall Framework-改.png" width="860" />

## Generative User Modeling with Chain-of-Thought

We expect the LLM to infer and integrate the three different aspects of user modeling. To achieve this, we employ chain-of-thought prompting in four logical steps as follows, reasoning about the user's attributes, preferences, and intentions.

<img src="https://github.com/kwyyangfan/GUMRec/blob/master/docs/CoT-GUM-V3.png" width="860" />

## Prompt Construction for Recommendation Tasks

We design a collection of prompts to align the LLMs with various recommendation tasks, encompassing rating prediction, sequential recommendation, direct recommendation, and explanation generation.

<img src="https://github.com/kwyyangfan/GUMRec/blob/master/docs/Prompt examples for four tasks-改.png" width="860" />

## Code Usage<a name="code" />

### Requirement<a name="requirement" />

``` bash 
conda create -n gumrec python=3.8
```

``` bash
# CUDA 10.2
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

```bash
pip install -r requirements.txt
```
### LLMs<a name="llm" />

Evaluate with OpenAI [GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5) and ZhipuAI [ChatGLM2-6B](https://github.com/THUDM/ChatGLM-6B)


----------
### Evaluating with ChatGLM2-6B<a name="ChatGLM" />

Run the scripts:

```bash
python generate.py -k <chatglm_key> -d [movie|book|beauty]
```

Indicating your ChatGLM key. 


----------

### Evaluating with GPT-3.5<a name="GPT" />

Go to the utils-evaluate fold, and run the metrics scripts:

```bash
python metrics4rec.py -k <openai_key> -d [movie|book|beauty]
```

Indicating your openai key. 
  
## Datasets Acquire

BaiduCloud: <https://pan.baidu.com/s/14ndz_wFpoHU0o5hCYp45lQ?pwd=qbt2> code: qbt2
