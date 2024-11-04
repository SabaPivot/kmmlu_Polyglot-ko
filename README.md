## KMMLU 데이터를 학습하여 추론

### 사용 데이터셋 및 모델
*dataset: HAERAE-HUB/KMMLU* <br>
*model: EleutherAI/polyglot-ko-1.3b*


### data.py
ㄴ four_to_choice: **네 개의 선택지를 하나의 리스트로 합쳐서 반환**
ㄴ polyglot_prompt: **Instruction Prompt 템플릿**
ㄴ tokenize_function: **훈련을 위한 tokenizing config setting**
