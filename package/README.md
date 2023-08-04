# PEFT-SER [[Paper Link](https://arxiv.org/abs/2306.05350)] was accepted to 11th International Conference on Affective Computing and Intelligent Interaction (ACII), 2023. 

#### This work include implementations for PEFT-SER: On the Use of Parameter Efficient Transfer Learning Approaches For Speech Emotion Recognition Using Pre-trained Speech Models. PEFT-SER is an open source project for researchers exploring SER applications with using parameter efficient finetuning methods.


#### This python package provides checkpoints that was trained on the combined data of IEMOCAP, MSP-Improv, MSP-Podcast, and CREMA-D, using the following models that shows promising results:

1. Whisper Tiny
2. Whisper Base
3. Whisper Small
4. Massively Multilingual Speech (MMS)
5. WavLM Base+
6. WavLM Large


#### We provide the checkpoints with LoRa rank of 16 according to our paper, as LoRa in general provides the best performance in PEFT SER. You can start to use our model within 5 lines of code. The package is under license, and it does not support commercial use.


### 1. Installation
```
pip install peft-ser
```

### 2. Model Loading
```
# whisper style loading
import peft_ser
model = peft_ser.load_model("whisper-base-lora-16-conv")

data = torch.zeros([1, 16000])
output = model(data)
```
#### a. Output mapping
The output emotion mappings are: **{0: "Neutral", 1: "Angry", 2: "Sad", 3: "Happy"}**. We would add a version for 6-emotion later.

#### b. Training details
For all the released models, we train/evaluate with the same data.  Unlike the ACII paper where the audio was restricted to 6s, these open-release models support the audio duration to the maximum of 10s for broader use cases. We also combine the convolutional output along with the transformer encodings for fine-tuning, as we find this further increase the model performance. We used a fix seed of 8, training epoch of 30, and learning rate of 2.5x10e-4.

#### c. Training/validation/test splits for reproducing the results

**The validation set:** Session 4 of IEMOCAP and Session 4 of MSP-Improv dataset, Validation set of MSP-Podcast dataset, and Speaker 1059-1073

**The evaluation set:** Session 5 of IEMOCAP and Session 5 of MSP-Improv dataset, Test set of MSP-Podcast dataset, and Speaker 1074-1091

**All rest data are used for training.**

#### d. Performance

Pre-trained Model | Test Performance without PEFT | Test Performance with LoRa | PEFT Model Name
|---|---|---|---
Whisper Tiny | 62.26 | 63.48 | whisper-tiny-lora-16-conv 
Whisper Base | 64.39 | 64.92 | whisper-base-lora-16-conv 
Whisper Small | 65.77 | 66.01 | whisper-small-lora-16-conv 
MMS |  |  | mms-lora-16-conv 
WavLM Base+ | 63.06 | 66.11 | wavlm-plus-lora-16-conv 
WavLM Large | 68.54 | **68.66** | wavlm-large-lora-16-conv 

#### e. You are free to explore the use of existing models to further fine-tune on other SER datasets, more challenging tasks like 6-emotion, 8-emotion recognition, and also transfer learning on Arousal/Valence/Dominance.

### Citation

Please cite the following that includes the foundation of the code/methods used for experiments.

**[PEFT-SER](https://arxiv.org/abs/2306.05350)**
```
@article{feng2023peft,
  title={PEFT-SER: On the Use of Parameter Efficient Transfer Learning Approaches For Speech Emotion Recognition Using Pre-trained Speech Models},
  author={Feng, Tiantian and Narayanan, Shrikanth},
  journal={arXiv preprint arXiv:2306.05350},
  year={2023}
}
```
**[Trust-SER](https://arxiv.org/abs/2305.11229)**
```
@article{feng2023trustser,
  title={TrustSER: On the trustworthiness of fine-tuning pre-trained speech embeddings for speech emotion recognition},
  author={Feng, Tiantian and Hebbar, Rajat and Narayanan, Shrikanth},
  journal={arXiv preprint arXiv:2305.11229},
  year={2023}
}
```

You should also cite all the datasets being used including IEMOCAP, MSP-Improv, MSP-Podcast, and CREMA-D.

