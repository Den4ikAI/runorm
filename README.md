# RUNorm - Нормализатор русского текста перед синтезом речи

RUNorm - это библиотека для нормализации русского текста, написанная на языке Python. Она предназначена для подготовки текст перед синтезом речи.

## Особенности

- Нормализация чисел: преобразование числовых значений в текстовую форму.
- Нормализация сокращений: расшифровка и замена сокращений полными формами.
- Кириллизация: преобразование латинских символов в соответствующие кириллические.
- Озвучка аббревиатур: конвертирует аббревиатуру в побуквенный вариант. (GPT -> джи пи ти)

## Установка

```
pip install runorm
```

## Использование

Пример использования RUNorm:

```python
from runorm import RUNorm

# Используйте load(workdir="./local_cache") для кэширования моделей в указанной папке.
# Доступные модели: small, medium, big
# Выбирайте устройство используемое pytorch с помощью переменной device
normalizer = RUNorm()
normalizer.load(model_size="small", device="cpu")

while True:
    text = input(":> ")
    normalized_text = normalizer.norm(text)
    print(">>>", normalized_text)
```

## Модели

RUNorm предоставляет несколько предобученных моделей разного размера:
- `small`: маленькая модель для быстрой нормализации. Охватывает самые популярные кейсы. Базируется на FRED-T5-95M
- `medium`: средняя модель для баланса между скоростью и качеством. Базируется на ruT5-base (222M)
- `big`: большая модель для лучшего качества нормализации. Базируется на FRED-T5-Large (860M)

Вы можете выбрать подходящую модель при вызове метода `load()`.



## Лицензия

Этот проект распространяется под лицензией [Apache2.0 License](LICENSE).

## Контакты

Если у вас есть вопросы или предложения, пожалуйста, свяжитесь с автором проекта:
- TG: [ССЫЛКА](https://t.me/chckdskeasfsd)
- HuggingFace проекта: [HF](https://huggingface.co/RUNorm)

Будем рады вашим отзывам и сотрудничеству!

## Донат
Вы можете поддержать проект деньгами. Это поможет быстрее разрабатывать более качественные новые версии. 
CloudTips: https://pay.cloudtips.ru/p/b9d86686
