# app_langchain.py - World Listening & Wild Listening (LangChain版)

import gradio as gr
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import json
import asyncio
import time
import os
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Tuple, Optional

# LangChain imports
try:
    from langchain_openai import ChatOpenAI
    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_community.llms import Ollama
    from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain.schema import BaseMessage, HumanMessage, SystemMessage
    from langchain.callbacks import get_openai_callback
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("LangChainライブラリが見つかりません。pip install langchain langchain-openai langchain-anthropic langchain-google-genaiを実行してください。")

@dataclass
class HumanPersona:
    """人間ペルソナプロファイル"""
    id: int
    age: int
    gender: str
    country: str
    occupation: str
    education: str
    income_level: str
    family_status: str
    language: str
    urban_rural: str
    continent: str

@dataclass
class AnimalPersona:
    """動物ペルソナプロファイル"""
    id: int
    species: str
    habitat: str
    size_category: str
    diet_type: str
    activity_pattern: str
    social_structure: str
    lifespan_category: str
    conservation_status: str
    continent: str

class WorldDemographicsDB:
    """世界人口動態データベース"""
    
    def __init__(self):
        self.setup_world_demographics()
    
    def setup_world_demographics(self):
        """世界人口動態データの初期化"""
        
        # 年齢分布（世界平均）
        self.age_distribution = {
            (0, 14): 25.4, (15, 24): 15.5, (25, 34): 17.2, (35, 44): 13.8,
            (45, 54): 11.9, (55, 64): 8.7, (65, 74): 5.2, (75, 100): 2.3
        }
        
        # 国別人口分布（人口上位国＋その他）
        self.country_distribution = {
            '中国': 17.8, 'インド': 17.7, 'アメリカ': 4.2, 'インドネシア': 3.4,
            'パキスタン': 2.8, 'ブラジル': 2.7, 'ナイジェリア': 2.6, 'バングラデシュ': 2.1,
            'ロシア': 1.9, 'メキシコ': 1.6, '日本': 1.6, 'フィリピン': 1.4,
            'エチオピア': 1.4, 'ベトナム': 1.2, 'エジプト': 1.3, 'トルコ': 1.1,
            'イラン': 1.1, 'ドイツ': 1.1, 'タイ': 0.9, 'イギリス': 0.9,
            'その他': 32.8
        }
        
        # 主要言語分布
        self.language_distribution = {
            '中国語（標準）': 14.1, 'ヒンディー語': 6.0, '英語': 5.1, 'スペイン語': 4.9,
            'アラビア語': 4.2, 'ベンガル語': 3.3, 'ポルトガル語': 2.9, 'ロシア語': 2.2,
            '日本語': 1.7, 'フランス語': 1.3, 'ドイツ語': 1.0, '韓国語': 0.9,
            'ベトナム語': 0.9, 'トルコ語': 0.8, 'イタリア語': 0.7, 'その他': 50.0
        }
        
        # 世界職業分布
        self.occupation_distribution = {
            '農業・畜産業': 26.2, 'サービス業': 15.8, '製造業': 12.6,
            '商業・貿易': 11.0, '建設業': 6.9, '教育関係': 4.7,
            '医療・介護': 4.2, '公務員': 3.8, 'IT・技術': 2.9,
            '運輸業': 2.8, '金融業': 2.1, '学生': 4.5, '無職': 2.5
        }
        
        # 教育レベル
        self.education_distribution = {
            '無学歴': 13.2, '初等教育': 28.4, '中等教育': 35.7,
            '職業訓練': 8.9, '高等教育': 11.2, '大学院': 2.6
        }
        
        # 年収分布（USD）
        self.income_distribution = {
            '1,000ドル未満': 15.3, '1,000-5,000ドル': 28.7, '5,000-15,000ドル': 26.9,
            '15,000-30,000ドル': 14.2, '30,000-50,000ドル': 7.8, '50,000-75,000ドル': 4.1,
            '75,000-100,000ドル': 1.8, '100,000ドル以上': 1.2
        }
        
        # 家族構成
        self.family_status_distribution = {
            '独身': 22.8, '既婚': 45.3, '既婚・子供あり': 25.5,
            'ひとり親': 4.2, '大家族': 2.2
        }

class TerrestrialAnimalDB:
    """陸上動物データベース"""
    
    def __init__(self):
        self.setup_animal_demographics()
    
    def setup_animal_demographics(self):
        """動物データの初期化"""
        
        # 種別分布（主要陸上動物）
        self.species_distribution = {
            'アフリカゾウ': 2.1, 'ライオン': 1.8, 'トラ': 0.8, 'ヒグマ': 2.3,
            'オオカミ': 3.2, 'アカギツネ': 4.5, 'シカ': 5.8, 'イノシシ': 3.9,
            'チンパンジー': 1.2, 'ゴリラ': 0.6, 'オランウータン': 0.4, 'ヒョウ': 1.5,
            'チーター': 0.3, 'キリン': 1.1, 'シマウマ': 2.7, 'サイ': 0.5,
            'カバ': 1.3, 'カンガルー': 3.4, 'パンダ': 0.2, 'ユキヒョウ': 0.3,
            'ジャガー': 0.7, 'ピューマ': 1.9, 'オオヤマネコ': 1.4, 'バイソン': 1.6,
            'ヘラジカ': 2.1, 'トナカイ': 3.8, 'ヤマヤギ': 1.7, 'その他': 53.5
        }
        
        # 生息環境
        self.habitat_distribution = {
            '熱帯雨林': 18.2, '温帯林': 15.4, '草原・サバンナ': 22.1,
            '砂漠': 8.7, '山岳地帯': 12.3, 'ツンドラ': 6.8, '湿地': 7.2,
            '沿岸部': 4.1, '混合環境': 5.2
        }
        
        # サイズカテゴリ
        self.size_distribution = {
            '極小（1kg未満）': 8.2, '小型（1-10kg）': 25.4, '中型（10-50kg）': 28.7,
            '大型（50-200kg）': 22.1, '超大型（200-1000kg）': 12.8, '巨大（1000kg以上）': 2.8
        }
        
        # 食性
        self.diet_distribution = {
            '草食動物': 42.3, '肉食動物': 18.7, '雑食動物': 28.4, '昆虫食': 10.6
        }
        
        # 活動パターン
        self.activity_distribution = {
            '昼行性': 45.2, '夜行性': 32.1, '薄明活動': 15.7, '無日周性': 7.0
        }
        
        # 社会構造
        self.social_distribution = {
            '単独行動': 38.4, 'つがい': 12.6, '小グループ': 23.8, '大群': 18.2, '複雑な社会': 7.0
        }
        
        # 寿命カテゴリ
        self.lifespan_distribution = {
            '短命（1-5年）': 15.2, '普通（5-15年）': 35.8, '長寿（15-30年）': 28.4,
            '非常に長寿（30-50年）': 15.3, '超長寿（50年以上）': 5.3
        }
        
        # 保護状況
        self.conservation_distribution = {
            '軽度懸念': 45.2, '準絶滅危惧': 18.7, '危急': 15.8,
            '絶滅危惧': 12.3, '深刻な危機': 8.0
        }

class PersonaGenerator:
    """人間・動物両対応ペルソナ生成器"""
    
    def __init__(self, mode: str):
        self.mode = mode
        if mode == "humans":
            self.db = WorldDemographicsDB()
        else:  # animals
            self.db = TerrestrialAnimalDB()
    
    def generate_weighted_choice(self, distribution: Dict[str, float]) -> str:
        """重み付き確率選択"""
        choices = list(distribution.keys())
        weights = list(distribution.values())
        return random.choices(choices, weights=weights)[0]
    
    def get_continent_from_country(self, country: str) -> str:
        """国から大陸を取得"""
        continent_map = {
            '中国': 'アジア', 'インド': 'アジア', 'アメリカ': '北米',
            'インドネシア': 'アジア', 'パキスタン': 'アジア', 'ブラジル': '南米',
            'ナイジェリア': 'アフリカ', 'バングラデシュ': 'アジア', 'ロシア': 'ヨーロッパ・アジア',
            'メキシコ': '北米', '日本': 'アジア', 'フィリピン': 'アジア',
            'エチオピア': 'アフリカ', 'ベトナム': 'アジア', 'エジプト': 'アフリカ',
            'トルコ': 'ヨーロッパ・アジア', 'イラン': 'アジア', 'ドイツ': 'ヨーロッパ',
            'タイ': 'アジア', 'イギリス': 'ヨーロッパ'
        }
        return continent_map.get(country, '世界各地')
    
    def get_continent_from_habitat(self, habitat: str) -> str:
        """生息環境から大陸を推定"""
        habitat_continent_map = {
            '熱帯雨林': 'アフリカ・南米・アジア',
            '温帯林': '北米・ヨーロッパ',
            '草原・サバンナ': 'アフリカ・アジア',
            '砂漠': 'アフリカ・アジア・オーストラリア',
            '山岳地帯': '世界各地',
            'ツンドラ': '北米・ヨーロッパ・アジア',
            '湿地': '世界各地',
            '沿岸部': '世界各地',
            '混合環境': '世界各地'
        }
        return habitat_continent_map.get(habitat, '世界各地')
    
    def generate_human_persona(self, persona_id: int) -> HumanPersona:
        """人間ペルソナ生成"""
        # 年齢生成
        age_ranges = list(self.db.age_distribution.keys())
        age_weights = list(self.db.age_distribution.values())
        selected_range = random.choices(age_ranges, weights=age_weights)[0]
        age = random.randint(selected_range[0], selected_range[1])
        
        # 基本属性
        country = self.generate_weighted_choice(self.db.country_distribution)
        
        persona = HumanPersona(
            id=persona_id,
            age=age,
            gender=random.choice(['男性', '女性']),
            country=country,
            occupation=self.generate_weighted_choice(self.db.occupation_distribution),
            education=self.generate_weighted_choice(self.db.education_distribution),
            income_level=self.generate_weighted_choice(self.db.income_distribution),
            family_status=self.generate_weighted_choice(self.db.family_status_distribution),
            language=self.generate_weighted_choice(self.db.language_distribution),
            urban_rural=random.choice(['都市部', '地方']),
            continent=self.get_continent_from_country(country)
        )
        
        return persona
    
    def generate_animal_persona(self, persona_id: int) -> AnimalPersona:
        """動物ペルソナ生成"""
        habitat = self.generate_weighted_choice(self.db.habitat_distribution)
        
        persona = AnimalPersona(
            id=persona_id,
            species=self.generate_weighted_choice(self.db.species_distribution),
            habitat=habitat,
            size_category=self.generate_weighted_choice(self.db.size_distribution),
            diet_type=self.generate_weighted_choice(self.db.diet_distribution),
            activity_pattern=self.generate_weighted_choice(self.db.activity_distribution),
            social_structure=self.generate_weighted_choice(self.db.social_distribution),
            lifespan_category=self.generate_weighted_choice(self.db.lifespan_distribution),
            conservation_status=self.generate_weighted_choice(self.db.conservation_distribution),
            continent=self.get_continent_from_habitat(habitat)
        )
        
        return persona
    
    def generate_persona(self, persona_id: int):
        """モードに応じたペルソナ生成"""
        if self.mode == "humans":
            return self.generate_human_persona(persona_id)
        else:
            return self.generate_animal_persona(persona_id)

class LangChainCostTracker:
    """LangChain用コスト追跡クラス"""
    
    def __init__(self):
        self.total_cost = 0.0
        self.total_tokens = 0
        self.requests_count = 0
        self.provider_costs = {}
        
    def add_openai_callback_result(self, result):
        """OpenAIコールバック結果を追加"""
        self.total_cost += result.total_cost
        self.total_tokens += result.total_tokens
        self.requests_count += 1
        
        if 'openai' not in self.provider_costs:
            self.provider_costs['openai'] = {'cost': 0, 'tokens': 0, 'requests': 0}
        
        self.provider_costs['openai']['cost'] += result.total_cost
        self.provider_costs['openai']['tokens'] += result.total_tokens
        self.provider_costs['openai']['requests'] += 1
    
    def add_manual_cost(self, cost: float, tokens: int, provider: str):
        """手動でコストを追加（他のプロバイダー用）"""
        self.total_cost += cost
        self.total_tokens += tokens
        self.requests_count += 1
        
        if provider not in self.provider_costs:
            self.provider_costs[provider] = {'cost': 0, 'tokens': 0, 'requests': 0}
        
        self.provider_costs[provider]['cost'] += cost
        self.provider_costs[provider]['tokens'] += tokens
        self.provider_costs[provider]['requests'] += 1
    
    def get_cost_summary(self) -> Dict:
        """コストサマリー取得"""
        return {
            'total_cost_usd': self.total_cost,
            'total_cost_jpy': self.total_cost * 150,
            'total_tokens': self.total_tokens,
            'requests_count': self.requests_count,
            'cost_per_request': self.total_cost / max(self.requests_count, 1),
            'provider_breakdown': self.provider_costs
        }

class LangChainLLMProvider:
    """LangChain用LLMプロバイダー"""
    
    def __init__(self, provider_type: str, api_key: str, model_name: str = None):
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChainライブラリが必要です")
        
        self.provider_type = provider_type
        self.cost_tracker = LangChainCostTracker()
        
        # プロバイダー別のLLM初期化
        if provider_type == "openai":
            self.llm = ChatOpenAI(
                api_key=api_key,
                model=model_name or "gpt-4o-mini",
                temperature=0.7,
                max_tokens=150
            )
        elif provider_type == "anthropic":
            self.llm = ChatAnthropic(
                api_key=api_key,
                model=model_name or "claude-3-haiku-20240307",
                temperature=0.7,
                max_tokens=150
            )
        elif provider_type == "google":
            self.llm = ChatGoogleGenerativeAI(
                api_key=api_key,
                model=model_name or "gemini-pro",
                temperature=0.7
            )
        elif provider_type == "ollama":
            self.llm = Ollama(
                model=model_name or "llama2",
                temperature=0.7
            )
        else:
            raise ValueError(f"サポートされていないプロバイダー: {provider_type}")
        
        # プロンプトテンプレートの設定
        self.setup_prompt_templates()
        
        # チェーンの作成
        self.setup_chains()
    
    def setup_prompt_templates(self):
        """プロンプトテンプレートの設定"""
        
        # 人間用システムプロンプト
        self.human_system_template = SystemMessagePromptTemplate.from_template(
            """あなたは{country}出身の{age}歳の{gender}です。
あなたの背景情報:
- 職業: {occupation}
- 教育: {education}
- 言語: {language}
- 家族構成: {family_status}
- 住環境: {urban_rural}

あなたの文化的背景、価値観、生活経験を考慮して、質問に自然に回答してください。
150文字以内で、この人物らしい声で答えてください。"""
        )
        
        # 動物用システムプロンプト
        self.animal_system_template = SystemMessagePromptTemplate.from_template(
            """あなたは{habitat}に住む{species}です。
あなたの特徴:
- サイズ: {size_category}
- 食性: {diet_type}
- 活動パターン: {activity_pattern}
- 社会構造: {social_structure}
- 保護状況: {conservation_status}

この動物の本能、生態、環境との関係を考慮して、質問に対するこの種ならではの反応を示してください。
150文字以内で、この動物の視点から答えてください。"""
        )
        
        # 人間用チャットプロンプト
        self.human_chat_template = ChatPromptTemplate.from_messages([
            self.human_system_template,
            HumanMessagePromptTemplate.from_template("{question}")
        ])
        
        # 動物用チャットプロンプト
        self.animal_chat_template = ChatPromptTemplate.from_messages([
            self.animal_system_template,
            HumanMessagePromptTemplate.from_template("{question}")
        ])
    
    def setup_chains(self):
        """チェーンの設定"""
        
        # 出力パーサー
        self.output_parser = StrOutputParser()
        
        # 人間用チェーン
        self.human_chain = self.human_chat_template | self.llm | self.output_parser
        
        # 動物用チェーン
        self.animal_chain = self.animal_chat_template | self.llm | self.output_parser
    
    async def generate_response(self, persona: Dict, question: str, mode: str) -> Dict:
        """LangChainを使用した回答生成"""
        
        try:
            # チェーンの選択
            if mode == "humans":
                chain = self.human_chain
                chain_input = {
                    "age": persona["age"],
                    "gender": persona["gender"],
                    "country": persona["country"],
                    "occupation": persona["occupation"],
                    "education": persona["education"],
                    "language": persona["language"],
                    "family_status": persona["family_status"],
                    "urban_rural": persona["urban_rural"],
                    "question": question
                }
            else:
                chain = self.animal_chain
                chain_input = {
                    "species": persona["species"],
                    "habitat": persona["habitat"],
                    "size_category": persona["size_category"],
                    "diet_type": persona["diet_type"],
                    "activity_pattern": persona["activity_pattern"],
                    "social_structure": persona["social_structure"],
                    "conservation_status": persona["conservation_status"],
                    "question": question
                }
            
            # OpenAIの場合はコストトラッキング
            if self.provider_type == "openai":
                with get_openai_callback() as cb:
                    response = await chain.ainvoke(chain_input)
                    self.cost_tracker.add_openai_callback_result(cb)
                    cost_usd = cb.total_cost
                    tokens_used = cb.total_tokens
            else:
                # 他のプロバイダーの場合（概算）
                response = await chain.ainvoke(chain_input)
                cost_usd = 0.0  # プロバイダー別のコスト計算をここに追加
                tokens_used = len(question) // 3 + len(response) // 3  # 概算
                self.cost_tracker.add_manual_cost(cost_usd, tokens_used, self.provider_type)
            
            return {
                'success': True,
                'response': response,
                'cost_usd': cost_usd,
                'tokens_used': tokens_used,
                'provider': self.provider_type
            }
            
        except Exception as e:
            return {
                'success': False,
                'response': f"エラー: {str(e)[:50]}...",
                'cost_usd': 0.0,
                'tokens_used': 0,
                'provider': self.provider_type,
                'error': str(e)
            }

class AdvancedAnalysisChain:
    """高度な分析用LangChainチェーン"""
    
    def __init__(self, llm_provider: LangChainLLMProvider):
        self.llm = llm_provider.llm
        self.setup_analysis_chains()
    
    def setup_analysis_chains(self):
        """分析用チェーンの設定"""
        
        # 洞察生成プロンプト
        self.insight_prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                """あなたは専門的な調査分析者です。
以下の調査結果データを分析し、主要な洞察と提言を生成してください。
データの傾向、パターン、意味のある発見を特定し、
実用的な提言を含めた包括的な分析を提供してください。"""
            ),
            HumanMessagePromptTemplate.from_template(
                """調査データ:
{survey_data}

質問: {question}

上記のデータから得られる主要な洞察と提言を生成してください。"""
            )
        ])
        
        # 洞察生成チェーン
        self.insight_chain = self.insight_prompt | self.llm | StrOutputParser()
    
    async def generate_insights(self, survey_data: str, question: str) -> str:
        """LLMを使用した洞察生成"""
        
        try:
            insights = await self.insight_chain.ainvoke({
                "survey_data": survey_data,
                "question": question
            })
            return insights
            
        except Exception as e:
            return f"洞察生成エラー: {str(e)}"

class SimulationProvider:
    """シミュレーション用プロバイダー（LangChain未使用時）"""
    
    def __init__(self, mode: str):
        self.mode = mode
        self.cost_tracker = LangChainCostTracker()
        
        # 回答パターン（前回と同じ）
        self.human_response_patterns = {
            '25歳未満': [
                "将来世代にとって本当に重要な問題だと思います。",
                "SNSで見た情報だと、早急に対策が必要そうです。",
                "今行動しないと手遅れになりそうで心配です。"
            ],
            '25-65歳': [
                "経験上、バランスの取れた解決策が必要だと思います。",
                "経済的な影響も考慮しつつ、環境対策を進めるべきです。",
                "家族の将来を考えると、真剣に取り組むべき課題です。"
            ],
            '65歳以上': [
                "長年の変化を見てきて、これは重要な問題です。",
                "孫の世代のことを考えると、対策が必要です。",
                "持続可能な解決策を見つけることが大切だと思います。"
            ]
        }
        
        self.animal_response_patterns = {
            '肉食動物': [
                "獲物の数が変わると、狩りがもっと大変になります。",
                "生息地の分断で、狩りの範囲が狭くなっています。",
                "気候の変化で、獲物がいる場所や時期が変わってきています。"
            ],
            '草食動物': [
                "植物の生える場所や時期が変わって、食べ物探しが難しくなっています。",
                "季節のずれで、好きな植物が食べられる時期が変わりました。",
                "人間の活動で、食べ物のある場所が減っています。"
            ],
            '雑食動物': [
                "食べ物の種類が多いので適応できるけど、だんだん厳しくなっています。",
                "植物も動物も食べられるから何とかなるけど、環境の変化は感じます。",
                "柔軟な食性が助かるけど、住む場所がなくなるのは困ります。"
            ]
        }
    
    async def generate_response(self, persona: Dict, question: str, mode: str) -> Dict:
        """シミュレーション回答生成"""
        await asyncio.sleep(0.1)
        
        if mode == "humans":
            age = persona.get('age', 30)
            if age < 25:
                age_group = '25歳未満'
            elif age <= 65:
                age_group = '25-65歳'
            else:
                age_group = '65歳以上'
            
            responses = self.human_response_patterns[age_group]
            
        else:
            diet_type = persona.get('diet_type', '雑食動物')
            responses = self.animal_response_patterns.get(diet_type, self.animal_response_patterns['雑食動物'])
        
        response = random.choice(responses)
        
        return {
            'success': True,
            'response': response,
            'cost_usd': 0.0,
            'tokens_used': 0,
            'provider': 'simulation'
        }

# エビデンスベース質問プリセット
EVIDENCE_BASED_QUESTIONS = {
    "気候変動の影響": "気候変動はあなたの環境や日常生活にどのような影響を与えていますか？",
    "生物多様性の保全": "生物多様性を守るために最も重要だと思う取り組みは何ですか？",
    "生息地の保護": "あなたの地域での生息地保護はどのくらい重要ですか？",
    "持続可能な実践": "みんなが取り組むべき持続可能な実践は何だと思いますか？",
    "種の絶滅への懸念": "種の絶滅についてどのくらい心配していますか？",
    "再生可能エネルギー": "将来、再生可能エネルギーはどのような役割を果たすべきですか？",
    "プラスチック汚染": "プラスチック汚染はあなたの環境にどのような影響を与えていますか？",
    "水資源の保全": "最も重要な水資源保全対策は何だと思いますか？",
    "森林破壊への対策": "森林破壊を止めるために何が必要だと思いますか？",
    "環境教育の重要性": "環境教育はどのくらい重要だと思いますか？"
}

# LLMプロバイダー設定
LLM_PROVIDERS = {
    "OpenAI GPT-4o-mini": {"type": "openai", "model": "gpt-4o-mini"},
    "OpenAI GPT-4": {"type": "openai", "model": "gpt-4"},
    "Anthropic Claude-3-Haiku": {"type": "anthropic", "model": "claude-3-haiku-20240307"},
    "Anthropic Claude-3-Sonnet": {"type": "anthropic", "model": "claude-3-sonnet-20240229"},
    "Google Gemini Pro": {"type": "google", "model": "gemini-pro"},
    "Ollama Llama2": {"type": "ollama", "model": "llama2"},
    "シミュレーション（無料）": {"type": "simulation", "model": None}
}

# グローバル状態管理
class AppState:
    def __init__(self):
        self.mode = "humans"
        self.personas = []
        self.survey_responses = []
        self.llm_provider = None
        self.analysis_chain = None
        self.selected_provider = "シミュレーション（無料）"

app_state = AppState()

# Gradioインターフェース関数群
def get_app_title():
    """アプリタイトル取得"""
    if app_state.mode == "humans":
        return "🌍 World Listening (LangChain版)"
    else:
        return "🦁 Wild Listening (LangChain版)"

def set_mode(mode):
    """調査モード設定"""
    app_state.mode = mode
    app_state.personas = []
    app_state.survey_responses = []
    return f"モード設定: {get_app_title()}"

def set_llm_provider(provider_name, api_key=""):
    """LLMプロバイダー設定"""
    app_state.selected_provider = provider_name
    provider_config = LLM_PROVIDERS[provider_name]
    
    if provider_config["type"] == "simulation":
        return "✅ シミュレーションモードが有効になりました（無料）"
    
    if not api_key:
        return f"❌ {provider_name}を使用するにはAPIキーが必要です"
    
    try:
        app_state.llm_provider = LangChainLLMProvider(
            provider_type=provider_config["type"],
            api_key=api_key,
            model_name=provider_config["model"]
        )
        
        # 高度分析チェーンの初期化
        app_state.analysis_chain = AdvancedAnalysisChain(app_state.llm_provider)
        
        return f"✅ {provider_name}プロバイダーの初期化が完了しました！"
        
    except Exception as e:
        return f"❌ {provider_name}の初期化エラー: {e}"

def generate_personas(num_personas=50):
    """ペルソナ生成（前回と同じ）"""
    try:
        generator = PersonaGenerator(app_state.mode)
        personas = []
        
        for i in range(num_personas):
            persona = generator.generate_persona(i + 1)
            personas.append(asdict(persona))
        
        app_state.personas = personas
        
        # サマリー作成
        df = pd.DataFrame(personas)
        if app_state.mode == "humans":
            summary = f"""
✅ {len(personas)}人の人間ペルソナを生成しました:
- 平均年齢: {df['age'].mean():.1f}歳
- 性別分布: {df['gender'].value_counts().to_dict()}
- 上位国家: {df['country'].value_counts().head(3).to_dict()}
- 言語数: {len(df['language'].unique())}種類の言語
"""
        else:
            summary = f"""
✅ {len(personas)}体の動物ペルソナを生成しました:
- 種数: {len(df['species'].unique())}種類の動物
- 生息環境分布: {df['habitat'].value_counts().head(3).to_dict()}
- 食性分布: {df['diet_type'].value_counts().to_dict()}
- 保護状況: {df['conservation_status'].value_counts().to_dict()}
"""
        
        return summary, create_persona_chart()
        
    except Exception as e:
        return f"❌ ペルソナ生成エラー: {e}", None

def create_persona_chart():
    """ペルソナ分布チャート作成（前回と同じ）"""
    if not app_state.personas:
        return None
    
    df = pd.DataFrame(app_state.personas)
    
    if app_state.mode == "humans":
        fig = px.histogram(df, x='age', title='年齢分布', nbins=20,
                          labels={'x': '年齢', 'y': '人数'})
    else:
        species_counts = df['species'].value_counts().head(10)
        fig = px.bar(x=species_counts.values, y=species_counts.index, 
                    orientation='h', title='上位10種の分布',
                    labels={'x': '個体数', 'y': '種名'})
    
    return fig

def run_survey(question, custom_question=""):
    """LangChainを使用した調査実行"""
    if not app_state.personas:
        return "❌ まずペルソナを生成してください", None, ""
    
    final_question = custom_question if custom_question.strip() else question
    if not final_question or final_question == "質問を選択してください...":
        return "❌ 質問を選択または入力してください", None, ""
    
    try:
        # プロバイダー初期化
        provider_config = LLM_PROVIDERS[app_state.selected_provider]
        
        if provider_config["type"] == "simulation":
            provider = SimulationProvider(app_state.mode)
        else:
            if not app_state.llm_provider:
                return "❌ LLMプロバイダーが初期化されていません", None, ""
            provider = app_state.llm_provider
        
        # 調査実行
        responses = []
        total_cost = 0
        
        async def run_async_survey():
            nonlocal responses, total_cost
            for persona in app_state.personas:
                result = await provider.generate_response(persona, final_question, app_state.mode)
                
                response = {
                    'persona_id': persona['id'],
                    'persona': persona,
                    'question': final_question,
                    'response': result['response'],
                    'success': result.get('success', True),
                    'cost_usd': result.get('cost_usd', 0.0),
                    'provider': result.get('provider', 'unknown'),
                    'timestamp': datetime.now().isoformat()
                }
                
                responses.append(response)
                total_cost += result.get('cost_usd', 0.0)
        
        # 非同期調査実行
        asyncio.run(run_async_survey())
        
        app_state.survey_responses = responses
        
        # サマリー作成
        successful_responses = len([r for r in responses if r['success']])
        summary = f"""
✅ 調査完了！
- 質問: {final_question}
- 総回答数: {len(responses)}
- 成功回答数: {successful_responses}
- 総コスト: ${total_cost:.6f} (約{total_cost * 150:.2f}円)
- プロバイダー: {app_state.selected_provider}
"""
        
        return summary, create_results_chart(), get_sample_responses()
        
    except Exception as e:
        return f"❌ 調査実行エラー: {e}", None, ""

def create_results_chart():
    """結果可視化作成（前回と同じ）"""
    if not app_state.survey_responses:
        return None
    
    response_lengths = [len(r['response']) for r in app_state.survey_responses]
    
    fig = px.histogram(x=response_lengths, title='回答長分布', 
                      labels={'x': '回答長（文字数）', 'y': '回答数'})
    
    return fig

def get_sample_responses():
    """サンプル回答取得（前回と同じ）"""
    if not app_state.survey_responses:
        return ""
    
    sample_size = min(5, len(app_state.survey_responses))
    sample_responses = random.sample(app_state.survey_responses, sample_size)
    
    output = "📝 回答サンプル:\n\n"
    
    for i, response in enumerate(sample_responses, 1):
        persona = response['persona']
        if app_state.mode == "humans":
            profile = f"{persona['country']}の{persona['age']}歳{persona['gender']}"
        else:
            profile = f"{persona['habitat']}の{persona['species']}"
        
        output += f"{i}. **{profile}**\n"
        output += f"💬 {response['response']}\n\n"
    
    return output

def generate_ai_insights():
    """AI洞察生成（LangChain使用）"""
    if not app_state.survey_responses:
        return "❌ まず調査を実行してください"
    
    if not app_state.analysis_chain:
        return "❌ 分析用LLMが初期化されていません"
    
    try:
        # 調査データの準備
        survey_data = ""
        for response in app_state.survey_responses[:10]:  # サンプル10件
            persona = response['persona']
            if app_state.mode == "humans":
                profile = f"{persona['country']}の{persona['age']}歳{persona['gender']}"
            else:
                profile = f"{persona['habitat']}の{persona['species']}"
            
            survey_data += f"- {profile}: {response['response']}\n"
        
        question = app_state.survey_responses[0]['question']
        
        # 非同期で洞察生成
        async def generate_async():
            return await app_state.analysis_chain.generate_insights(survey_data, question)
        
        insights = asyncio.run(generate_async())
        return f"🤖 AI生成洞察:\n\n{insights}"
        
    except Exception as e:
        return f"❌ AI洞察生成エラー: {e}"

def export_results():
    """結果CSV出力（前回と同じ）"""
    if not app_state.survey_responses:
        return None
    
    export_data = []
    for response in app_state.survey_responses:
        row = {
            'persona_id': response['persona_id'],
            'question': response['question'],
            'response': response['response'],
            'success': response['success'],
            'provider': response.get('provider', 'unknown'),
            'timestamp': response['timestamp']
        }
        
        persona = response['persona']
        for key, value in persona.items():
            row[f'persona_{key}'] = value
        
        export_data.append(row)
    
    df = pd.DataFrame(export_data)
    
    filename = f"langchain_survey_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')
    
    return filename

# Gradioインターフェース作成
def create_interface():
    """LangChain版Gradioインターフェース作成"""
    
    with gr.Blocks(title="World & Wild Listening - LangChain版", theme=gr.themes.Soft()) as demo:
        
        # 動的タイトル
        title_display = gr.Markdown("# 🌍 World Listening (LangChain版)")
        
        gr.Markdown("""
        ### 🚀 LangChain搭載 - マルチLLM対応世界規模調査システム
        
        **新機能:**
        - 🔄 複数LLMプロバイダー対応（OpenAI、Anthropic、Google、Ollama）
        - 🧠 AI洞察自動生成
        - 📊 高度な統計分析
        - 💰 プロバイダー別コスト最適化
        """)
        
        with gr.Tabs():
            # 設定タブ
            with gr.Tab("⚙️ 設定"):
                gr.Markdown("### 調査対象選択")
                
                mode_radio = gr.Radio(
                    choices=[("人間（世界）", "humans"), ("動物（陸上）", "animals")],
                    value="humans",
                    label="調査対象",
                    info="人間ペルソナまたは動物ペルソナのどちらを調査するかを選択"
                )
                
                mode_status = gr.Textbox(label="モード状況", value="モード: 🌍 World Listening (LangChain版)")
                
                gr.Markdown("### 🤖 LLMプロバイダー選択")
                
                provider_dropdown = gr.Dropdown(
                    choices=list(LLM_PROVIDERS.keys()),
                    value="シミュレーション（無料）",
                    label="LLMプロバイダー",
                    info="使用するLLMプロバイダーを選択"
                )
                
                api_key_input = gr.Textbox(
                    label="APIキー",
                    type="password",
                    placeholder="プロバイダーのAPIキーを入力...",
                    info="シミュレーション以外を選択した場合に必要"
                )
                
                llm_status = gr.Textbox(label="LLM状況", value="シミュレーションモード（無料）")
                
                # イベントハンドラー
                def update_title_and_mode(mode):
                    app_state.mode = mode
                    if mode == "humans":
                        title = "# 🌍 World Listening (LangChain版)"
                        status = "モード: 🌍 World Listening (LangChain版)"
                    else:
                        title = "# 🦁 Wild Listening (LangChain版)"
                        status = "モード: 🦁 Wild Listening (LangChain版)"
                    return title, status
                
                mode_radio.change(
                    fn=update_title_and_mode,
                    inputs=[mode_radio],
                    outputs=[title_display, mode_status]
                )
                
                provider_dropdown.change(
                    fn=set_llm_provider,
                    inputs=[provider_dropdown, api_key_input],
                    outputs=[llm_status]
                )
                
                api_key_input.change(
                    fn=set_llm_provider,
                    inputs=[provider_dropdown, api_key_input],
                    outputs=[llm_status]
                )
            
            # ペルソナ生成タブ
            with gr.Tab("👥 ペルソナ"):
                gr.Markdown("### 調査ペルソナ生成")
                
                num_personas = gr.Slider(
                    minimum=10,
                    maximum=200,
                    value=50,
                    step=10,
                    label="ペルソナ数",
                    info="多いほど包括的な調査（実LLM使用時はコスト増）"
                )
                
                generate_btn = gr.Button("🎲 ペルソナ生成", variant="primary")
                
                persona_status = gr.Textbox(label="生成状況")
                persona_chart = gr.Plot(label="ペルソナ分布")
                
                generate_btn.click(
                    fn=generate_personas,
                    inputs=[num_personas],
                    outputs=[persona_status, persona_chart]
                )
            
            # 調査タブ
            with gr.Tab("❓ 調査"):
                gr.Markdown("### 🌱 エビデンスベース調査実行")
                
                question_dropdown = gr.Dropdown(
                    choices=["質問を選択してください..."] + list(EVIDENCE_BASED_QUESTIONS.keys()),
                    value="質問を選択してください...",
                    label="プリセット質問",
                    info="環境・保全・持続可能性に関するエビデンスベースのトピック"
                )
                
                custom_question = gr.Textbox(
                    label="カスタム質問",
                    placeholder="または独自の質問を入力...",
                    lines=3,
                    info="プリセット質問使用時は空のままで"
                )
                
                def update_custom_question(selected):
                    if selected and selected != "質問を選択してください...":
                        return EVIDENCE_BASED_QUESTIONS[selected]
                    return ""
                
                question_dropdown.change(
                    fn=update_custom_question,
                    inputs=[question_dropdown],
                    outputs=[custom_question]
                )
                
                run_survey_btn = gr.Button("🚀 調査実行", variant="primary")
                
                survey_status = gr.Textbox(label="調査状況")
                results_chart = gr.Plot(label="結果可視化")
                sample_responses = gr.Textbox(
                    label="サンプル回答",
                    lines=10,
                    max_lines=20
                )
                
                run_survey_btn.click(
                    fn=run_survey,
                    inputs=[question_dropdown, custom_question],
                    outputs=[survey_status, results_chart, sample_responses]
                )
            
            # AI洞察タブ
            with gr.Tab("🧠 AI洞察"):
                gr.Markdown("### 🤖 LangChainによる高度な分析")
                
                gr.Markdown("""
                **機能:**
                - 調査結果の自動分析
                - パターン認識と洞察生成
                - 実用的な提言の生成
                - 多角的視点での解釈
                """)
                
                generate_insights_btn = gr.Button("🧠 AI洞察生成", variant="primary")
                
                ai_insights = gr.Textbox(
                    label="AI生成洞察",
                    lines=15,
                    max_lines=30
                )
                
                generate_insights_btn.click(
                    fn=generate_ai_insights,
                    outputs=[ai_insights]
                )
            
            # エクスポートタブ
            with gr.Tab("📤 エクスポート"):
                gr.Markdown("### 結果出力")
                
                export_btn = gr.Button("📊 CSVエクスポート", variant="secondary")
                export_file = gr.File(label="結果ダウンロード")
                
                export_btn.click(
                    fn=export_results,
                    outputs=[export_file]
                )
                
                gr.Markdown("""
                ### エクスポート内容
                CSVファイルには以下が含まれます：
                - 全調査回答
                - 完全なペルソナプロファイル
                - 使用プロバイダー情報
                - タイムスタンプとメタデータ
                - 成功・失敗ステータス
                """)
        
        # フッター
        gr.Markdown("""
        ---
        **🚀 Powered by LangChain** | マルチLLM対応・コスト最適化
        
        **対応プロバイダー:**
        - 🔵 OpenAI (GPT-4o-mini, GPT-4)
        - 🟣 Anthropic (Claude-3-Haiku, Claude-3-Sonnet)
        - 🔴 Google (Gemini Pro)
        - 🟢 Ollama (Llama2, その他ローカルモデル)
        
        **エビデンスベーストピック:**
        - 気候変動と環境影響
        - 生物多様性と保全
        - 持続可能な実践
        - 環境教育
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )