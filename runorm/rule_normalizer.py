import re
from .number_to_word import Numbers2Words
class RuleNormalizer:
    def __init__(self):
        self.pronunciation_map = {
            'А': 'а', 'Б': 'бэ', 'В': 'вэ', 'Г': 'гэ', 'Д': 'дэ',
            'Е': 'е', 'Ё': 'ё', 'Ж': 'жэ', 'З': 'зэ', 'И': 'и',
            'Й': 'ий', 'К': 'ка', 'Л': 'эл', 'М': 'эм', 'Н': 'эн',
            'О': 'о', 'П': 'пэ', 'Р': 'эр', 'С': 'эс', 'Т': 'тэ',
            'У': 'у', 'Ф': 'эф', 'Х': 'ха', 'Ц': 'цэ', 'Ч': 'чэ',
            'Ш': 'ша', 'Щ': 'ща', 'Ъ': 'твёрдый знак', 'Ы': 'ы', 'Ь': 'мягкий знак',
            'Э': 'э', 'Ю': 'ю', 'Я': 'я'
        }
        self.cyrrilization_mapping_extended = {
            'A': 'А', 'B': 'Б', 'C': 'К', 'D': 'Д', 'E': 'Е',
            'F': 'Ф', 'G': 'Г', 'H': 'Х', 'I': 'И', 'J': 'Й',
            'K': 'К', 'L': 'Л', 'M': 'М', 'N': 'Н', 'O': 'О',
            'P': 'П', 'Q': 'К', 'R': 'Р', 'S': 'С', 'T': 'Т',
            'U': 'У', 'V': 'В', 'W': 'В', 'X': 'КС', 'Y': 'Ы',
            'Z': 'З',
            'a': 'а', 'b': 'б', 'c': 'к', 'd': 'д', 'e': 'е',
            'f': 'ф', 'g': 'г', 'h': 'х', 'i': 'и', 'j': 'й',
            'k': 'к', 'l': 'л', 'm': 'м', 'n': 'н', 'o': 'о',
            'p': 'п', 'q': 'к', 'r': 'р', 's': 'с', 't': 'т',
            'u': 'у', 'v': 'в', 'w': 'в', 'x': 'кс', 'y': 'ы',
            'z': 'з',
            # Common digraphs
            'SH': 'Ш', 'CH': 'Ч', 'TH': 'З', 'PH': 'Ф', 'OO': 'У', 'EE': 'И', 'KH': 'Х',
            'sh': 'ш', 'ch': 'ч', 'th': 'з', 'ph': 'ф', 'oo': 'у', 'ee': 'и', 'kh': 'х',
            # common trigraphs
            'SCH': 'СК',
            'sch': 'ск'
            # Capital letters are also converted to lowercase in the cyrrilization
        }
        self.numbers_normalizer = Numbers2Words()


    def cyrrilize(self, text):
        cyrrilized_text = ""
        i = 0
        while i < len(text):
            if i + 1 < len(text) and text[i:i+2] in self.cyrrilization_mapping_extended:
                # If a digraph is found, add its cyrrilization and increment by 2
                cyrrilized_text += self.cyrrilization_mapping_extended[text[i:i+2]]
                i += 2
            else:
                # Add the cyrrilization of a single character
                cyrrilized_text += self.cyrrilization_mapping_extended.get(text[i], text[i])
                i += 1
        return cyrrilized_text

    def expand_abbreviations(self, text):
        abbreviations = re.findall(r'\b[А-ЯЁ]{2,}\b', text)

        for abbr in abbreviations:
            pronounced_form = ' '.join(self.pronunciation_map[letter] for letter in abbr if letter in self.pronunciation_map)
            text = text.replace(abbr, pronounced_form)

        return text

    def detect_numbers(self, text):
        number_pattern = re.compile(r'\b\d+\b')
        matches = list(number_pattern.finditer(text))
        number_matches = [{'number': match.group(), 'start': match.start(), 'end': match.end()} for match in matches]
    
        return number_matches

    def number_to_words_digit_by_digit(self, n):
        units = ['ноль', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять']
        return ' '.join(units[int(digit)] for digit in str(n))

    def normalize_text_with_numbers(self, text):
        detected_numbers = self.detect_numbers(text)
        detected_numbers.sort(key=lambda x: x['start'], reverse=True)
    
        for num in detected_numbers:
            number_value = int(num['number'])
            if number_value >= 1_000_000_000_000:
                normalized_number = self.number_to_words_digit_by_digit(number_value)
            else:
                normalized_number = sself.numbers_normalizer.numbers_to_words(number_value)
            text = text[:num['start']] + normalized_number + text[num['end']:]

        return text

    def normalize_phone_number(self, phone_number):
        digits = re.sub(r'\D', '', phone_number)

        segments = {
            'country_code': digits[:1],
            'area_code': digits[1:4],
            'block_1': digits[4:7],
            'block_2': digits[7:9],
            'block_3': digits[9:11],
        }

        if segments['country_code'] == '8':
            segments['country_code'] = 'восемь'
        elif segments['country_code'] == '7':
            segments['country_code'] = 'плюс семь'

        normalized_segments = {
            key: self.numbers_normalizer.numbers_to_words(int(value)) if key != 'country_code' else value
            for key, value in segments.items()
        }

        spoken_form = ' '.join(normalized_segments.values())

        return spoken_form

    def normalize_text_with_phone_numbers(self, text):
        phone_pattern = re.compile(
            r"(?:\+7)\s*\(?\d{3}\)?\s*\d{3}[-\s]?\d{2}[-\s]?\d{2}|8\d{10}"
        )
        matches = list(phone_pattern.finditer(text))
        detected_phone_numbers = [{'phone': match.group().strip(), 'start': match.start(), 'end': match.end()} for match in matches]

        detected_phone_numbers.sort(key=lambda x: x['start'], reverse=True)
    
        for pn in detected_phone_numbers:
            normalized_phone = self.normalize_phone_number(pn['phone'])
            text = text[:pn['start']] + normalized_phone + text[pn['end']:]

        return text

    def currency_normalization(self, text):
        def russian_plural(number, units):
            if number % 10 == 1 and number % 100 != 11:
                return units[0]
            elif 2 <= number % 10 <= 4 and (number % 100 < 10 or number % 100 >= 20):
                return units[1]
            else:
                return units[2]

        def currency_to_words(amount, currency='rub'):
            currencies = {
                'rub': (['рубль', 'рубля', 'рублей'], ['копейка', 'копейки', 'копеек']),
                'usd': (['доллар', 'доллара', 'долларов'], ['цент', 'цента', 'центов']),
                'eur': (['евро', 'евро', 'евро'], ['евроцент', 'евроцента', 'евроцентов']),
                'gbp': (['фунт', 'фунта', 'фунтов'], ['пенс', 'пенса', 'пенсов']),
                'uah': (['гривна', 'гривны', 'гривен'], ['копейка', 'копейки', 'копеек']),
            }

            main_units, sub_units = currencies.get(currency, currencies['rub'])

            main_amount = int(amount)
            sub_amount = int(round((amount - main_amount) * 100))
            main_words = self.numbers_normalizer.numbers_to_words(main_amount) + ' ' + russian_plural(main_amount, main_units)
            
            sub_words = ''

            if sub_amount > 0:
                sub_words = self.numbers_normalizer.numbers_to_words(sub_amount) + ' ' + russian_plural(sub_amount, sub_units)

            full_currency_words = main_words.strip()
            if sub_words:
                full_currency_words += ' ' + sub_words.strip()

            return full_currency_words

        currency_patterns = {
            'rub': [r'(\d+(?:\.\d\d)?)\s*(руб(л(ей|я|ь))?|₽)', r'(\d+(?:\.\d\d)?)\s*RUB'],
            'usd': [r'(\d+(?:\.\d\d)?)\s*(доллар(ов|а|ы)?|\$)', r'(\d+(?:\.\d\d)?)\s*USD', r'\$(\d+(?:\.\d\d)?)'],
            'eur': [r'(\d+(?:\.\d\d)?)\s*(евро|€)', r'(\d+(?:\.\d\d)?)\s*EUR', r'(\d+)\s*€'],
            'gbp': [r'(\d+(?:\.\d\d)?)\s*(фунт(ов|а|ы)?|£)', r'(\d+(?:\.\d\d)?)\s*GBP', r'£(\d+)'],
            'uah': [r'(\d+(?:\.\d\d)?)\s*(грив(ен|ны|на)|₴)', r'(\d+(?:\.\d\d)?)\s*UAH', r'(\d+)\s*₴']
        }

        def detect_currency(text):
            for currency_code, patterns in currency_patterns.items():
                for pattern in patterns:
                    matches = re.finditer(pattern, text)
                    for match in matches:
                        amount = float(match.group(1))
                        currency_words = currency_to_words(amount, currency_code)
                        text = re.sub(pattern, currency_words, text, count=1)

            return text

        return detect_currency(text)

    def normalize(self, text):
        #text = self.cyrrilize(text)
        text = self.expand_abbreviations(text)
        text = self.currency_normalization(text)
        text = self.normalize_text_with_phone_numbers(text)
        return text