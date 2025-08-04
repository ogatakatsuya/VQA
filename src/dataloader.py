import random
import re
from statistics import mode

import torch
import pandas
import numpy as np
from qwen_vl_utils import process_vision_info


def set_seed(seed):
    """
    シードを固定する．

    Parameters
    ----------
    seed : int
        乱数生成に用いるシード値．
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_text(text) -> str:
    """
    入力文と回答のフォーマットを統一するための関数．

    Parameters
    ----------
    text : str
        入力文，もしくは回答．
    """
    # lowercase
    text = text.lower()

    # 数詞を数字に変換
    num_word_to_digit = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }
    for word, digit in num_word_to_digit.items():
        text = text.replace(word, digit)

    # 小数点のピリオドを削除
    text = re.sub(r"(?<!\d)\.(?!\d)", "", text)

    # 冠詞の削除
    text = re.sub(r"\b(a|an|the)\b", "", text)

    # 短縮形のカンマの追加
    contractions = {
        "dont": "don't",
        "isnt": "isn't",
        "arent": "aren't",
        "wont": "won't",
        "cant": "can't",
        "wouldnt": "wouldn't",
        "couldnt": "couldn't",
    }
    for contraction, correct in contractions.items():
        text = text.replace(contraction, correct)

    # 句読点をスペースに変換
    text = re.sub(r"[^\w\s':]", " ", text)

    # 句読点をスペースに変換
    text = re.sub(r"\s+,", ",", text)

    # 連続するスペースを1つに変換
    text = re.sub(r"\s+", " ", text).strip()

    return text


class VQADataset(torch.utils.data.Dataset):
    """
    VQA データセットを扱うためのクラス．
    """

    def __init__(
        self,
        df_path,
        image_dir,
        has_answer=True,
    ):
        self.image_dir = image_dir
        self.df = pandas.read_json(df_path)
        self.has_answer = has_answer

    def __getitem__(self, idx):
        image_path = f"{self.image_dir}/{self.df['image'][idx]}"
        question = self.df["question"][idx]

        if self.has_answer:
            answers = [
                process_text(answer["answer"]) for answer in self.df["answers"][idx]
            ]
            mode_answer = mode(answers)

            return (
                image_path,
                question,
                mode_answer,
            )
        else:
            return image_path, question

    def __len__(self):
        return len(self.df)


def collate_fn(batch, processor):
    is_train = len(batch[0]) == 3

    if is_train:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": question},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": answer}]},
            ]
            for image_path, question, answer in batch
        ]
    else:
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]
            for image_path, question in batch
        ]

    text = processor.apply_chat_template(
        messages,
        add_generation_prompt=(not is_train),
        tokenize=False,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=text,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    return inputs
