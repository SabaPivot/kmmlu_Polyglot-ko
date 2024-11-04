def four_to_choice(data) -> dict[str, list]:
    """
    A, B, C, D 선택지를 하나의 choices(list)로 합쳐서 넣어주는 함수
    """
    choices = [data['A'], data['B'], data['C'], data['D']]
    return {'choices': choices}


def polyglot_prompt(data, if_train):
    prompt_template = """당신은 AI 챗봇입니다. 사용자에게 도움이 되고 유익한 내용을 제공해야합니다. 답변은 길고 자세하며 친절한 설명을 덧붙여서 작성하세요.

### 사용자:
문제:
{question}

선택지:
{choices}

### 챗봇:
답변:
{answer}"""

    # Directly access question, choices, and answer
    question = data["question"].strip()
    choices = data["choices"]
    answer = data["answer"] if if_train else ""  # Only include answer if if_train is True
    choices_str = '\n'.join(choices).strip()

    # Format the prompt
    text = prompt_template.format(
        question=question,
        choices=choices_str,
        answer=answer
    ) + ("<|endoftext|>" if if_train else "")

    return {"text": text}  # Return a dictionary with text list

def tokenize_function(examples, tokenizer):
    result = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=2048,
        return_token_type_ids=False
    )
    # Set labels equal to input_ids and mask padding tokens
    result["labels"] = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in result["input_ids"]
    ]
    return result
