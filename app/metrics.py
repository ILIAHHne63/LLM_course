import re
from transformers import pipeline
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


def parse_original_news(original_input: str) -> List[Dict[str, str]]:
    """
    Парсит оригинальный формат новостей и извлекает тексты.
    
    Args:
        original_input: Строка с запросом и новостями в формате:
            "Извлеките информацию из следующих новостных сообщений по запросу /{запрос}/:
             Сообщение 1:
             - Дата и время: /{дата и время}/
             - Текст: "/{текст новости}/"
             ..."
    
    Returns:
        List[Dict]: Список словарей с информацией о новостях
    """
    news_list = []
    
    # Разделяем по "Сообщение N:"
    messages = re.split(r'Сообщение \d+:', original_input)
    
    for i, message in enumerate(messages[1:], 1):  # Пропускаем первый элемент (запрос)
        # Извлекаем дату и время
        date_match = re.search(r'- Дата и время:\s*(.+)', message)
        date_time = date_match.group(1).strip() if date_match else "Не указано"
        
        # Извлекаем текст новости (между кавычками или после "- Текст:")
        text_match = re.search(r'- Текст:\s*["\']?(.+?)["\']?\s*(?:Сообщение|$)', message, re.DOTALL)
        if not text_match:
            text_match = re.search(r'- Текст:\s*(.+)', message, re.DOTALL)
        
        if text_match:
            news_text = text_match.group(1).strip().strip('"\'')
            news_list.append({
                'id': i,
                'date_time': date_time,
                'text': news_text
            })
    
    return news_list


def parse_transformed_news(transformed_input: str) -> List[Optional[str]]:
    """
    Парсит преобразованные новости в новом упрощённом формате.
    
    Args:
        transformed_input: Строка с преобразованными новостями в формате:
            "Сообщение 1: "/{текст новости}/"
             Сообщение 2: "/{текст новости}/"
             ..."
    
    Returns:
        List[Optional[str]]: Список преобразованных текстов новостей (None если "-")
    """
    news_texts = []
    
    # Разделяем по "Сообщение N:"
    messages = re.split(r'Сообщение \d+:', transformed_input)
    
    for message in messages[1:]:  # Пропускаем первый элемент, если есть
        # Очищаем текст
        message = message.strip()
        
        # Извлекаем текст, убирая кавычки и лишние символы
        # Ищем текст в кавычках или просто первый текст
        text_match = re.search(r'["\']?(.+?)["\']?\s*(?:Сообщение|$)', message, re.DOTALL)
        
        if text_match:
            news_text = text_match.group(1).strip().strip('"\'')
            
            # Проверяем, является ли текст прочерком (неподтвержденная новость)
            if news_text.strip() == '-':
                news_texts.append(None)
            else:
                news_texts.append(news_text)
        else:
            news_texts.append(None)
    
    return news_texts


def classify_text(text: str, classifier) -> Tuple[str, float, float]:
    """
    Классифицирует текст как субъективный или объективный.
    
    Args:
        text: Текст для классификации
        classifier: Pipeline для классификации
    
    Returns:
        Tuple[str, float, float]: (метка в русском виде, уверенность, объективность_скор)
            - объективность_скор: 0.0 (полностью субъективный) до 1.0 (полностью объективный)
    """
    result = classifier(text)[0]
    label = result['label']
    score = result['score']
    
    # Преобразуем метку в русский формат
    if label in ['LABEL_1']:
        label_ru = "Субъективная"
        objectivity_score = 1.0 - score  # Низкая объективность
    else:
        label_ru = "Объективная"
        objectivity_score = score  # Высокая объективность
    
    return label_ru, score, objectivity_score


def evaluate_subjectivity_filtering_extended(
    original_input: str,
    transformed_input: str
) -> Tuple[float, float, float, int, float, float]:
    """
    Оценивает качество фильтрации оценочных суждений из новостей с расширенными метриками.
    
    Args:
        original_input: Оригинальные новости с запросом в формате:
            "Извлеките информацию из следующих новостных сообщений по запросу /{запрос}/:
             Сообщение 1: ..."
        transformed_input: Преобразованные новости в упрощенном формате:
            "Сообщение 1: ..."
    
    Returns:
        Tuple[float, float, float, int, float, float]: 
            - Средняя оценка объективности оригинальных новостей (без неподтвержденных)
            - Средняя оценка объективности преобразованных новостей (без неподтвержденных)
            - Средняя оценка объективности неподтвержденных новостей
            - Количество неподтвержденных новостей
            - Доля новостей SUBJ → OBJ
            - Доля новостей OBJ → SUBJ
    """

    subjectivity_classifier = pipeline(
    "text-classification",
    model="GroNLP/mdebertav3-subjectivity-multilingual",
    device=0
)
    
    # Парсинг входных данных
    original_news = parse_original_news(original_input)
    
    transformed_news = parse_transformed_news(transformed_input)
    
    # Проверка соответствия количества
    if len(original_news) != len(transformed_news):
        # Выравниваем списки
        max_len = max(len(original_news), len(transformed_news))
        while len(original_news) < max_len:
            original_news.append({'id': len(original_news)+1, 'date_time': 'N/A', 'text': ''})
        while len(transformed_news) < max_len:
            transformed_news.append(None)
        print()
    
    # Инициализация списков для метрик
    original_objectivity_scores = []
    transformed_objectivity_scores = []
    unconfirmed_objectivity_scores = []
    
    results = []
    
    for i, (orig_news, trans_text) in enumerate(zip(original_news, transformed_news), 1):
        
        # Классификация оригинальной новости
        orig_label, orig_score, orig_obj_score = classify_text(
            orig_news['text'], subjectivity_classifier
        )
        
        # Проверка на неподтвержденную новость
        if trans_text is None:
            unconfirmed_objectivity_scores.append(orig_obj_score)
            
            results.append({
                'original_label': orig_label,
                'original_score': orig_score,
                'original_objectivity': orig_obj_score,
                'transformed_label': None,
                'transformed_score': None,
                'transformed_objectivity': None,
                'is_unconfirmed': True
            })
        else:
            # Классификация преобразованной новости
            trans_label, trans_score, trans_obj_score = classify_text(
                trans_text, subjectivity_classifier
            )
            
            original_objectivity_scores.append(orig_obj_score)
            transformed_objectivity_scores.append(trans_obj_score)
            
            results.append({
                'original_label': orig_label,
                'original_score': orig_score,
                'original_objectivity': orig_obj_score,
                'transformed_label': trans_label,
                'transformed_score': trans_score,
                'transformed_objectivity': trans_obj_score,
                'is_unconfirmed': False
            })
        
        print()
    
    # 1. Средняя оценка объективности оригинальных новостей (без неподтвержденных)
    avg_orig_objectivity = (
        sum(original_objectivity_scores) / len(original_objectivity_scores)
        if original_objectivity_scores else 0.0
    )
    
    # 2. Средняя оценка объективности преобразованных новостей
    avg_trans_objectivity = (
        sum(transformed_objectivity_scores) / len(transformed_objectivity_scores)
        if transformed_objectivity_scores else 0.0
    )
    
    # 3. Средняя оценка объективности неподтвержденных новостей
    avg_unconf_objectivity = (
        sum(unconfirmed_objectivity_scores) / len(unconfirmed_objectivity_scores)
        if unconfirmed_objectivity_scores else 0.0
    )
    
    # 4. Количество неподтвержденных новостей
    num_unconfirmed = len(unconfirmed_objectivity_scores)
    
    # 5. Доля субъективных → объективных (только для подтвержденных)
    confirmed_results = [r for r in results if not r['is_unconfirmed']]
    
    subj_to_obj = sum(
        1 for r in confirmed_results 
        if r['original_label'] == 'Субъективная' and r['transformed_label'] == 'Объективная'
    )
    total_original_subj = sum(
        1 for r in confirmed_results if r['original_label'] == 'Субъективная'
    )
    
    ratio_subj_to_obj = subj_to_obj / total_original_subj if total_original_subj > 0 else 0.0
    
    # 6. Доля объективных → субъективных
    obj_to_subj = sum(
        1 for r in confirmed_results 
        if r['original_label'] == 'Объективная' and r['transformed_label'] == 'Субъективная'
    )
    total_original_obj = sum(
        1 for r in confirmed_results if r['original_label'] == 'Объективная'
    )
    
    ratio_obj_to_subj = obj_to_subj / total_original_obj if total_original_obj > 0 else 0.0

    metrics_dict = {"avg_orig_objectivity": avg_orig_objectivity, "avg_trans_objectivity": avg_trans_objectivity, "avg_unconf_objectivity": avg_unconf_objectivity, "num_unconfirmed": num_unconfirmed, "ratio_subj_to_obj": ratio_subj_to_obj, "ratio_obj_to_subj": ratio_obj_to_subj}
    
    return metrics_dict