# Symco LLM Project Toolkit
This is the public version of a toolkit designed to perform Symmetry of Coordination (SYMCO) experimentation and data extraction on an expanding range of LLM's. It is being developed under the supervision of Dr. Adam Przepiorkowski in conjunction with the Univeristy of Warsaw and IPI PAN. 

# Goals
The overarching aim of the project is to expand ongoing research into SYMCO conducted with human participants into the realm of LLMs. This LLM subproject looks to run the same test questionnaire with pre-trained open-weight models such as LLaMA and Mistral, and extract the underlying 'thought' process revealed by weights and output probabilities (logits). 

# Execution
After several attempts to run the experiments using various API-served LLMS, It was ultimately determined that running models locally was the only way to ensure consistent results. The most recent iteration uses a dockerized RunPod deployment on 2 NVIDIA A100 SXMs. Theoretically, with additional funding for more cloud storage and computing power, this pipeline could be used to run any open-access model.

## Target Models
GPT-2 family

Meta LLaMA family

Mistral/Mixtral family

Qwen family

BLOOM family

# Sources
Code: Abraham Wilkins
Literature: 
On the Symmetry of Coordination
and Gradient Selectional Restrictions, Przepiorkowski and Patejuk (2025)

How Furiously Can Colorless Green Ideas Sleep?
Sentence Acceptability in Context, Lau. et al. 

Special thanks to Monika Gurak and Berke Sensekerci for assistance and feedback