import json
import re
import torch
from transformers import AutoTokenizer, GPT2Tokenizer, T5ForConditionalGeneration, BertForTokenClassification
from transformers import pipeline
from .number_to_word import Numbers2Words
import numpy as np
import time
from .rule_normalizer import RuleNormalizer
from razdel import sentenize
import pathlib



class RUNorm:
    def __init__(self):
        self.numbers_tokenizer = None
        self.numbers_model = None
        self.abbr_tokenizer = None
        self.abbr_model = None
        self.rule_normalizer = RuleNormalizer()
        self.numbers_normalizer = Numbers2Words()
        self.re_tokens = re.compile(r"(?:[.,!?]|[а-яА-Я]\S*|-?\d\S*(?:\.\d+)?|[^а-яА-Я\d\s-]+)\s*")
        self.re_normalization = re.compile(r"[^a-zA-Z0-9\sа-яА-ЯёЁ.,!?:;""''(){}\[\]«»„“”-]")
        self.paths = {
            "tagger": "RUNorm/RUNorm-tagger",
            "kirillizator": "RUNorm/RUNorm-kirillizator",
            "small": "RUNorm/RUNorm-normalizer-small",
            "medium": "RUNorm/RUNorm-normalizer-medium",
            "big": "RUNorm/RUNorm-normalizer-big"
        }
        
        
    def load(self, model_size="small", device="cpu", workdir=None):
        if workdir:
            self.workdir = workdir
        else:
            self.workdir = str(pathlib.Path(__file__).resolve().parent) + "/cache"

        self.model_size = model_size
        self.abbr_tokenizer = AutoTokenizer.from_pretrained(self.paths[model_size], cache_dir=self.workdir)
        self.abbr_model = T5ForConditionalGeneration.from_pretrained(self.paths[model_size], cache_dir=self.workdir)
        self.angl_tokenizer = AutoTokenizer.from_pretrained(self.paths["kirillizator"], cache_dir=self.workdir)
        self.angl_model = T5ForConditionalGeneration.from_pretrained(self.paths["kirillizator"], cache_dir=self.workdir)
        self.tagger_model = BertForTokenClassification.from_pretrained(self.paths["tagger"], cache_dir=self.workdir)
        self.tagger_tokenizer = AutoTokenizer.from_pretrained(self.paths["tagger"], cache_dir=self.workdir)
        self.tagger = pipeline("ner", model=self.tagger_model, tokenizer=self.tagger_tokenizer, aggregation_strategy="average")
        self.abbr_model.to(device)
        self.angl_model.to(device)
        self.abbr_model.eval()
        self.angl_model.eval()

    def normalize_input(self, text):
        return re.sub(self.re_normalization, "", text)

    def normalize_digits(self, text):
        pattern = r'(\d+)(\w+)'
        normalized_text = re.sub(pattern, r'\1 \2', text)
        return normalized_text

    def split_by_sentences(self, text):
        text = list(sentenize(text))
        text = [_.text for _ in text]
        return text

    def kirillizator(self, text):
        latin_to_cyrillic = {
            'a': 'эй', 'b': 'би', 'c': 'си', 'd': 'ди', 'e': 'и', 'f': 'эф',
            'g': 'джи', 'h': 'эйч', 'i': 'ай', 'j': 'джей', 'k': 'кей', 'l': 'эль',
            'm': 'эм', 'n': 'эн', 'o': 'оу', 'p': 'пи', 'q': 'кью', 'r': 'ар',
            's': 'эс', 't': 'ти', 'u': 'ю', 'v': 'ви', 'w': 'дабл-ю', 'x': 'экс',
            'y': 'уай', 'z': 'зэд'
        }
    
        text = text.lower()
        result = ""
    
        for char in text:
            if char.isalpha():
                result += latin_to_cyrillic.get(char, char)
            else:
                result += char
        result = result.strip()
        return result

    
    def process_sentence(self, text):
        return re.findall(self.re_tokens, text)

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    def is_english(self, word):
        english_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,!?:;-\'\"()[]{}/ '
        for char in word:
            if char not in english_chars:
                return False
        
        return True
    def construct_prompt(self, text, angl_mode=False):
        used = False
        result = ""
        etid = 0
        token_to_add = ""
        for token in self.process_sentence(text) + [""]:
            if not re.search("[a-zA-Z\d]", token):
                if token_to_add:
                    end_match = re.search(r"(.+?)(\W*)$", token_to_add, re.M).groups()
                    if self.is_english(end_match[0].strip()):
                        if angl_mode:
                            result += self.predict_anglicizms(end_match[0].strip()) + " "
                        else:
                            result += token_to_add
                    elif self.is_number(end_match[0].strip()):
                        normalized_number = self.numbers_normalizer.numbers_to_words(end_match[0].replace(' ',''))
                        if normalized_number.startswith("минус "):
                            result += "минус [{0}]<extra_id_{1}>{2}".format(normalized_number[6:], etid, end_match[1])
                            used = True
                        else:
                            result += f"[{normalized_number}]<extra_id_{etid}>{end_match[1]}"
                            used = True

                    else:
                        tags = self.tagger(token_to_add)
                        tags = self.process_tags(tags)
                        if tags[0]["entity_group"] == "TIME" or tags[0]["entity_group"] == "ORDINAL" or tags[0]["entity_group"] == "CARDINAL":
                            result += f"[{normalized_number if self.is_number(end_match[0].strip()) else end_match[0]}]<extra_id_{etid}>{end_match[1]}"
                            used = True
                        #pass
                        else:
                            result += token_to_add
                    etid += 1
                    token_to_add = ""
                    
                result += token
            else:
                token_to_add += token
        return result, used

    def construct_answer(self, prompt, prediction):
        re_prompt = re.compile(r"\[([^\]]+)\]<extra_id_(\d+)>")
        re_pred = re.compile(r"\<extra_id_(\d+)\>(.+?)(?=\<extra_id_\d+\>|</s>)")
        pred_data = {}
        for match in re.finditer(re_pred, prediction.replace("\n", " ")):
            pred_data[match[1]] = match[2].strip()
        while True:
            match = re.search(re_prompt, prompt)
            if not match:
                break
            replace = pred_data.get(match[2], match[1])
            prompt = prompt[:match.span()[0]] + replace + prompt[match.span()[1]:]
        return prompt

    def predict_abbr(self, prompt):
        if self.model_size != "medium":
            prompt = "<SC1>" + prompt
        data = self.abbr_tokenizer(prompt, return_tensors="pt")
        data = {k: v.to(self.abbr_model.device) for k, v in data.items()}
        output_ids = self.abbr_model.generate(
            **data, do_sample=False, max_new_tokens=512, repetition_penalty=1.0
        )[0]
        out = self.abbr_tokenizer.decode(output_ids.tolist())
        out = out.replace("<s>", "").replace("<pad>", "")
        return out

    def predict_anglicizms(self, prompt):
          data = self.angl_tokenizer("<pad><pad><pad>"+prompt, return_tensors="pt")
          data.pop("token_type_ids")
          data = {k: v.to(self.angl_model.device) for k, v in data.items()}
          output_ids = self.angl_model.generate(
              **data,  do_sample=False, max_new_tokens=128, repetition_penalty=1.0
          )[0]
          out = self.angl_tokenizer.decode(output_ids.tolist())
          out = out.replace("<s>","").replace("</s>","").replace("<pad>","").strip()
          out = self.kirillizator(out)
          return out

    def generate_text(self, model, tokenizer, input_text, max_length=100, temperature=0, top_k=0, top_p=1.0, repetition_penalty=1.0, skip_sp_tokens=False):
        input_ids = tokenizer.encode(input_text, return_tensors="pt")
        output_ids = torch.zeros((1, max_length), dtype=torch.long)
        output_ids[0, 0] = model.config.decoder_start_token_id

        for i in range(1, max_length):
            model_outputs = model(input_ids=input_ids, decoder_input_ids=output_ids[:, :i])
            logits = model_outputs.logits[:, -1, :]

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float("Inf")

            input_tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            output_tokens = tokenizer.convert_ids_to_tokens(output_ids[0, :i])
            for token in set(output_tokens):
                if token in input_tokens:
                    logits[:, tokenizer.convert_tokens_to_ids(token)] /= repetition_penalty

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            output_ids[0, i] = next_token

            if next_token == tokenizer.eos_token_id:
                break

        return tokenizer.decode(output_ids[0], skip_special_tokens=skip_sp_tokens)

    def count_vowels(self, s):
        vowels = "аеёиоуыэюя"
        count = 0
        for char in s.lower():
            if char in vowels:
                count += 1
        return count

    def process_tags(self, tags):
        processed_tags = []
        prev_tag = None
        for tag in tags:
            if tag['entity_group'] == "ABBREVIATION" and tag['word'] == ".":
                tag['entity_group'] = "PLAIN"
        for tag in tags:
            if tag['entity_group'] == "PLAIN" and tag['word'] == ".":
                tag['entity_group'] = prev_tag['entity_group']

            if prev_tag is None or tag['entity_group'] != prev_tag['entity_group']:
                processed_tags.append(tag)
            else:
                processed_tags[-1]['word'] += tag['word']
                processed_tags[-1]['end'] = tag['end']

            prev_tag = tag

        for entity in processed_tags:
            if entity['entity_group'] == "ABBREVIATION" and self.count_vowels(entity["word"]) > 1:
                entity['entity_group'] = "PLAIN"

        return processed_tags

    def proccess_abbr(self, text):
        tags = self.tagger(text)
        #print(tags)
        for tag in tags:
            if tag["entity_group"] != "PLAIN" and tag["entity_group"] != "TIME" and tag["entity_group"] != "PLAIN":
                tag["entity_group"] = "TAG"
        tags = self.process_tags(tags)
        prompt = ""
        current_index = 0
        extra_id_counter = 0
        used = False
        for entity in tags:
            if entity['entity_group'] == "TAG":
                prompt += text[current_index:entity['start']] + "[" + entity['word'] + "]<extra_id_" + str(extra_id_counter) + ">"
                extra_id_counter += 1
                used = True
            else:
                prompt += text[current_index:entity['start']] + entity['word']
            current_index = entity['end']
        
        prompt += text[current_index:]
        if used:
            result = self.predict_abbr(prompt)
            result = self.construct_answer(prompt, result)
        else:
            result = text
        return result
    def norm(self, message):
        #start = time.time()
        sentences = self.split_by_sentences(message)
        out = ""
        for message in sentences:
            message = self.rule_normalizer.normalize(message)
            message = message.capitalize()
            prompt, used = self.construct_prompt(message)
            if used:
                numbers_prediction = self.predict_abbr(prompt)
                answer = self.construct_answer(prompt, numbers_prediction)
                #print(prompt)
                #print(numbers_prediction)
            elif used is None:
                answer = prompt
            else:
                answer = message
                numbers_prediction = "NOT USED!"

            #print(answer)
            final_answer = self.proccess_abbr(answer)
            final_answer, _ = self.construct_prompt(final_answer, angl_mode=True)
            out = out + " " + final_answer
            
        #elapsed_time = time.time() - start
        return out.strip()