import json
from typing import Dict, List, Any

from natasha import (
    Segmenter, MorphVocab, NewsEmbedding,
    NewsNERTagger, AddrExtractor, DatesExtractor,
    Doc
)

from model import predict_complaint, get_advice


SPECIALISTS_PATH = 'D:/JKH_Diplom/data/json_data/specialists.json'
DOCS_PATH = 'D:/JKH_Diplom/data/json_data/jkh.json'


def load_json_data():
    """Load problems and specialists data from JSON files"""
    try:
        with open(DOCS_PATH, 'r', encoding='utf-8') as f:
            problems_data = json.load(f)

        with open(SPECIALISTS_PATH, 'r', encoding='utf-8') as f:
            specialists_data = json.load(f)

        return problems_data, specialists_data
    except Exception as e:
        print(f"Error loading JSON data: {e}")
        return {}, []


problems_data, specialists_data = load_json_data()


class ImprovedComplaintAnalyzer:
    def __init__(self, problems_data, specialists_data):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        self.ner_tagger = NewsNERTagger(emb)
        self.addr_extractor = AddrExtractor(self.morph_vocab)
        self.dates_extractor = DatesExtractor(self.morph_vocab)

        self.problems_data = problems_data
        self.specialists_data = specialists_data

    def analyze_complaint(self, text: str) -> Dict[str, Any]:
        """Анализирует жалобу: NER + модель"""
        ner_result = self.ner_analysis(text)
        model_result = self.model_analysis(text)
        category = model_result['prediction'].get('category', 'не определена')
        specialists = self.find_specialists(category)
        recommendations = self.generate_recommendations(category, specialists, model_result)

        combined = self.combine_results(ner_result, model_result, category, recommendations)

        return {
            'ner_result': ner_result,
            'model_result': model_result,
            'rag_result': {
                'category': category,
                'specialists': specialists,
                'recommendations': recommendations
            },
            'combined_analysis': combined
        }

    def ner_analysis(self, text: str) -> Dict[str, Any]:
        """Извлечение сущностей с помощью Natasha"""
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_ner(self.ner_tagger)

        addresses = self.addr_extractor(text)
        dates = self.dates_extractor(text)

        return {
            'addresses': [str(addr.fact) for addr in addresses],
            'dates': [str(date.fact) for date in dates],
            'entities': [(span.text, span.type) for span in doc.spans]
        }

    def model_analysis(self, text: str) -> Dict[str, Any]:
        """Категория, эмоция и срочность только из вашей модели"""
        try:
            prediction = predict_complaint(text)
            advice = get_advice(prediction)
            return {
                'prediction': prediction,
                'advice': advice
            }
        except Exception as e:
            return {
                'prediction': {
                    'category': 'не определена',
                    'emotion': 'нейтральная',
                    'urgency': 'средняя'
                },
                'advice': 'Специалист свяжется с вами в ближайшее время.',
                'error': str(e)
            }

    def find_specialists(self, category: str) -> List[Dict]:
        """Ищет специалистов по категории"""
        required_specializations = self.determine_required_specializations(category)

        suitable_specialists = []
        for specialist in self.specialists_data:
            specializations = [s.lower() for s in specialist.get('specialization', [])]
            position = specialist.get('position', '').lower()

            matches = sum(
                1 for req in required_specializations
                if any(req in spec for spec in specializations) or req in position
            )

            if matches > 0:
                suitable_specialists.append({
                    'specialist': specialist,
                    'relevance': matches,
                    'district': specialist.get('district', 'Не указан')
                })

        suitable_specialists.sort(key=lambda x: x['relevance'], reverse=True)
        return suitable_specialists[:5]

    def determine_required_specializations(self, category: str) -> List[str]:
        """Специалисты по категориям"""
        mapping = {
            'gas_leak_emergency': ['аварийный', 'газ', 'сантехник'],
            'heating_system_break': ['отопление', 'сантехник', 'аварийный'],
            'electricity_supply_break': ['электрик', 'электричество', 'аварийный'],
            'sewage_system_break': ['сантехник', 'канализация', 'аварийный'],
            'water_supply_break': ['сантехник', 'вода', 'аварийный'],
            'elevator_breakdown': ['лифт', 'техник'],
            'roof_leak': ['кровля', 'кровельщик'],
            'noise_violations': ['диспетчер', 'координация'],
            'garbage_chute_problems': ['диспетчер', 'мусор', 'обслуживание']
        }
        return mapping.get(category, ['диспетчер'])

    def generate_recommendations(self, category: str, specialists: List[Dict], model_result: Dict) -> Dict[str, Any]:
        """Генерация рекомендаций на основе категории модели"""
        model_prediction = model_result.get('prediction', {})
        urgency = model_prediction.get('urgency', 'средняя')

        fallback_recommendations = {
            'gas_leak_emergency': {
                'actions': ['Немедленно покиньте помещение', 'Перекройте газ, если это безопасно', 'Вызовите аварийную газовую службу'],
                'contacts': ['104', '112'],
                'deadlines': 'немедленно'
            },
            'heating_system_break': {
                'actions': ['Проверьте батареи у соседей', 'Сообщите в УК', 'Вызовите сантехника'],
                'contacts': ['диспетчерская УК'],
                'deadlines': '1-3 часа'
            },
            'electricity_supply_break': {
                'actions': ['Проверьте автоматы в щитке', 'Сообщите в УК', 'Вызовите электрика'],
                'contacts': ['112', 'диспетчерская УК'],
                'deadlines': '2-4 часа'
            },
            'sewage_system_break': {
                'actions': ['Не пользуйтесь сливом', 'Сообщите в УК', 'Вызовите сантехника'],
                'contacts': ['диспетчерская УК'],
                'deadlines': '2-4 часа'
            },
            'water_supply_break': {
                'actions': ['Перекройте воду', 'Сообщите в УК', 'Вызовите сантехника'],
                'contacts': ['диспетчерская УК'],
                'deadlines': '4 часа'
            },
            'elevator_breakdown': {
                'actions': ['Не пользуйтесь лифтом', 'Сообщите в диспетчерскую', 'Ожидайте техника'],
                'contacts': ['диспетчерская лифта'],
                'deadlines': '1-2 часа'
            },
            'roof_leak': {
                'actions': ['Зафиксируйте протечку', 'Сообщите в УК', 'Ограничьте доступ к зоне протечки'],
                'contacts': ['диспетчерская УК'],
                'deadlines': '2-6 часов'
            },
            'noise_violations': {
                'actions': ['Зафиксируйте нарушение', 'Сообщите участковому или в УК', 'Попробуйте урегулировать конфликт'],
                'contacts': ['102', 'диспетчерская УК'],
                'deadlines': 'в рабочее время'
            }
        }

        fallback = fallback_recommendations.get(category, {
            'actions': ['Обратитесь в диспетчерскую службу'],
            'contacts': ['8-800-100-00-00'],
            'deadlines': 'не указаны'
        })

        return {
            'urgency': urgency,
            'actions': fallback['actions'],
            'deadlines': fallback['deadlines'],
            'contacts': fallback['contacts'],
            'recommended_specialists': [s['specialist']['name'] for s in specialists[:3]],
            'legal_basis': 'Жилищный кодекс РФ'
        }

    def combine_results(
        self,
        ner_result: Dict,
        model_result: Dict,
        category: str,
        recommendations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Собирает финальный результат"""
        return {
            'extracted_data': {
                'addresses': ner_result['addresses'],
                'dates': ner_result['dates'],
                'entities': ner_result['entities'],
                'additional_info': model_result
            },
            'problem_category': category,
            'urgency_level': model_result.get('prediction', {}).get('urgency', 'средняя'),
            'recommendations': recommendations
        }

    def generate_report(self, analysis_result: Dict) -> str:
        """Генерация отчета"""
        data = analysis_result['combined_analysis']
        recommendations = data['recommendations']
        category_display = data['problem_category'].replace('_', ' ').title()

        report = [
            "🔍 **ПОЛНЫЙ АНАЛИЗ ЖАЛОБЫ**",
            "",
            "📋 **ИЗВЛЕЧЕННЫЕ ДАННЫЕ:**",
            f"• 🏠 Адреса: {', '.join(data['extracted_data']['addresses']) or 'не найдены'}",
            f"• 📅 Даты: {', '.join(data['extracted_data']['dates']) or 'не найдены'}",
            f"• 🧩 Сущности: {', '.join([f'{t}({tp})' for t, tp in data['extracted_data']['entities']]) or 'не найдены'}",
            "",
            "📌 **РЕЗУЛЬТАТ МОДЕЛИ:**",
            f"• Категория: {analysis_result['model_result']['prediction'].get('category', 'не определена')}",
            f"• Эмоция: {analysis_result['model_result']['prediction'].get('emotion', 'нейтральная')}",
            f"• Срочность модели: {analysis_result['model_result']['prediction'].get('urgency', 'средняя')}",
            f"• Совет: {analysis_result['model_result'].get('advice', '')}",
            "",
            "📎 **ФИНАЛЬНАЯ КАТЕГОРИЯ:**",
            f"• {category_display}",
            "",
            "⚡ **РЕКОМЕНДАЦИИ:**",
            f"• Срочность: {recommendations['urgency']}",
            f"• Сроки решения: {recommendations['deadlines']}",
            "",
            "🛠 **НЕОБХОДИМЫЕ ДЕЙСТВИЯ:**"
        ]

        for i, action in enumerate(recommendations['actions'], 1):
            report.append(f"{i}. {action}")

        if recommendations['recommended_specialists']:
            report.extend([
                "",
                "👨‍💼 **РЕКОМЕНДУЕМЫЕ СПЕЦИАЛИСТЫ:**",
            ])
            for specialist in recommendations['recommended_specialists']:
                report.append(f"• {specialist}")

        report.extend([
            "",
            "📞 **ЭКСТРЕННЫЕ КОНТАКТЫ:**",
        ])

        for contact in recommendations['contacts']:
            report.append(f"• {contact}")

        if recommendations.get('legal_basis'):
            report.extend([
                "",
                "⚖️ **ПРАВОВАЯ БАЗА:**",
                recommendations['legal_basis']
            ])

        if data['extracted_data']['additional_info']:
            report.extend([
                "",
                "🤖 **ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:**"
            ])
            pred = data['extracted_data']['additional_info'].get('prediction', {})
            report.append(f"• category: {pred.get('category', 'не определена')}")
            report.append(f"• emotion: {pred.get('emotion', 'нейтральная')}")
            report.append(f"• urgency: {pred.get('urgency', 'средняя')}")
            if data['extracted_data']['additional_info'].get('advice'):
                report.append(f"• advice: {data['extracted_data']['additional_info']['advice']}")

        return "\n".join(report)


analyzer = ImprovedComplaintAnalyzer(problems_data, specialists_data)