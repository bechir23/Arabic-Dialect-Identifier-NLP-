# preprocessing.py
import re
import string
import os
import numpy as np
import pandas as pd
import random
import json
from pyarabic.araby import strip_tashkeel, strip_tatweel, strip_diacritics

# Preprocessor for MADAR Arabic dialect corpus
class MADARPreprocessor:
    """Preprocesses Arabic text for dialect ID using MADAR features."""
    
    def __init__(self, madar_lexicon_path=None):
        # Punctuation lists
        self.arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:\\\"؟.,'{}~¦+|!\\"…"–ـ'''
        self.english_punctuations = string.punctuation
        self.punctuations_list = self.arabic_punctuations + self.english_punctuations
        
        # Character normalization map
        self.char_map = {
            'إ': 'ا', 'أ': 'ا', 'آ': 'ا',
            'ة': 'ه',
            'ى': 'ي',
            'گ': 'ك',
            'ڤ': 'ف',
            'پ': 'ب',
            'چ': 'ج',
            'ژ': 'ز',
            'ڨ': 'ق',
            'ڭ': 'ك',
        }
        
        # Supported cities in MADAR
        self.cities = [
            'Cairo', 'Alexandria', 'Aswan',
            'Beirut', 'Damascus', 'Aleppo',
            'Jerusalem', 'Amman', 'Salt',
            'Doha', 'Riyadh', 'Jeddah', 'Muscat',
            'Rabat', 'Fes', 'Tripoli', 'Tunis',
            'Sfax', 'Algiers', 'Benghazi',
            'Baghdad', 'Basra', 'Mosul',
            'Khartoum', 'Sanaa'
        ]
        
        # City to region mapping
        self.city_to_region = {
            'Cairo': 'EGY', 'Alexandria': 'EGY', 'Aswan': 'EGY',
            'Beirut': 'LEV', 'Damascus': 'LEV', 'Aleppo': 'LEV',
            'Jerusalem': 'LEV', 'Amman': 'LEV', 'Salt': 'LEV',
            'Doha': 'GLF', 'Riyadh': 'GLF', 'Jeddah': 'GLF', 'Muscat': 'GLF',
            'Rabat': 'MGR', 'Fes': 'MGR', 'Tripoli': 'MGR', 'Tunis': 'MGR',
            'Sfax': 'MGR', 'Algiers': 'MGR', 'Benghazi': 'MGR',
            'Baghdad': 'IRQ', 'Basra': 'IRQ', 'Mosul': 'IRQ',
            'Khartoum': 'SDN',
            'Sanaa': 'YEM'
        }
        
        # Regions in MADAR
        self.regions = ['EGY', 'LEV', 'GLF', 'MGR', 'IRQ', 'SDN', 'YEM']
        
        # Initialize feature maps/dictionaries
        self.caphi_map = self._initialize_caphi_map()
        self.mwe_dict = self._initialize_mwe_dict()
        self.morph_features = self._initialize_morph_features()
        self.city_markers = self._initialize_city_markers()
        self.madar_lexicon = self._load_madar_lexicon(madar_lexicon_path)
        
    def _initialize_caphi_map(self):
        """Initialize CAPHI phonological map."""
        caphi_map = {}
        
        # ج (Jeem) pronunciation map
        caphi_map['ج'] = {
            'Cairo': 'g', 'Alexandria': 'g', 'Aswan': 'g',
            'Beirut': 'j', 'Damascus': 'j', 'Aleppo': 'j',
            'Jerusalem': 'j', 'Amman': 'j', 'Salt': 'j',
            'Doha': 'j', 'Riyadh': 'j', 'Jeddah': 'j',
            'Muscat': 'j',
            'Rabat': 'j', 'Fes': 'j', 'Tripoli': 'j',
            'Tunis': 'j', 'Sfax': 'j', 'Algiers': 'j',
            'Benghazi': 'j',
            'Baghdad': 'j', 'Basra': 'j', 'Mosul': 'j',
            'Khartoum': 'j',
            'Sanaa': 'j'
        }
        
        # ق (Qaf) pronunciation map
        caphi_map['ق'] = {
            'Cairo': "'", 'Alexandria': "'", 'Aswan': "'",
            'Beirut': "'", 'Damascus': "'", 'Aleppo': "'",
            'Jerusalem': "'", 'Amman': "'", 'Salt': "'",
            'Doha': 'g', 'Riyadh': 'g', 'Jeddah': 'g',
            'Muscat': 'g',
            'Rabat': 'q', 'Fes': 'q', 'Tripoli': 'q',
            'Tunis': 'q', 'Sfax': 'q', 'Algiers': 'q',
            'Benghazi': 'g',
            'Baghdad': 'q', 'Basra': 'q', 'Mosul': 'q',
            'Khartoum': 'g',
            'Sanaa': 'g'
        }
        
        # ث (Thaa) pronunciation map
        caphi_map['ث'] = {
            'Cairo': 's', 'Alexandria': 's', 'Aswan': 's',
            'Beirut': 's', 'Damascus': 's', 'Aleppo': 's',
            'Jerusalem': 's', 'Amman': 's', 'Salt': 'th',
            'Doha': 'th', 'Riyadh': 'th', 'Jeddah': 'th',
            'Muscat': 'th',
            'Rabat': 't', 'Fes': 't', 'Algiers': 't',
            'Tunis': 't', 'Sfax': 't', 'Tripoli': 't',
            'Benghazi': 't',
            'Baghdad': 'th', 'Basra': 'th', 'Mosul': 'th',
            'Khartoum': 't',
            'Sanaa': 'th'
        }
        
        # ذ (Thal) pronunciation map
        caphi_map['ذ'] = {
            'Cairo': 'z', 'Alexandria': 'z', 'Aswan': 'z',
            'Beirut': 'z', 'Damascus': 'z', 'Aleppo': 'z',
            'Jerusalem': 'z', 'Amman': 'z', 'Salt': 'dh',
            'Doha': 'dh', 'Riyadh': 'dh', 'Jeddah': 'dh',
            'Muscat': 'dh',
            'Rabat': 'd', 'Fes': 'd', 'Algiers': 'd',
            'Tunis': 'd', 'Sfax': 'd', 'Tripoli': 'd',
            'Benghazi': 'd',
            'Baghdad': 'dh', 'Basra': 'dh', 'Mosul': 'dh',
            'Khartoum': 'd',
            'Sanaa': 'dh'
        }
        
        # ظ (DHA) pronunciation map
        caphi_map['ظ'] = {
            'Cairo': 'z', 'Alexandria': 'z', 'Aswan': 'z',
            'Beirut': 'z', 'Damascus': 'z', 'Aleppo': 'z',
            'Jerusalem': 'z', 'Amman': 'z', 'Salt': 'dh',
            'Doha': 'dh', 'Riyadh': 'dh', 'Jeddah': 'dh',
            'Muscat': 'dh',
            'Rabat': 'd', 'Fes': 'd', 'Tripoli': 'd',
            'Tunis': 'd', 'Sfax': 'd', 'Algiers': 'd',
            'Benghazi': 'd',
            'Baghdad': 'dh', 'Basra': 'dh', 'Mosul': 'dh',
            'Khartoum': 'd',
            'Sanaa': 'dh'
        }
        
        # ك (Kaf) pronunciation map
        caphi_map['ك'] = {
            'Cairo': 'k', 'Alexandria': 'k', 'Aswan': 'k',
            'Beirut': 'k', 'Damascus': 'k', 'Aleppo': 'k',
            'Jerusalem': 'k', 'Amman': 'k', 'Salt': 'k',
            'Doha': 'ch', 'Riyadh': 'k', 'Jeddah': 'k',
            'Muscat': 'k',
            'Rabat': 'k', 'Fes': 'k', 'Tripoli': 'k',
            'Tunis': 'k', 'Sfax': 'k', 'Algiers': 'k',
            'Benghazi': 'k',
            'Baghdad': 'k', 'Basra': 'k', 'Mosul': 'k',
            'Khartoum': 'k',
            'Sanaa': 'k'
        }
        
        return caphi_map
        
    def _initialize_mwe_dict(self):
        """Initialize common multi-word expressions."""
        mwe_dict = {
            'عشان كده': 'عشان_كده',
            'مش عارف': 'مش_عارف',
            'على طول': 'على_طول',
            'شو في': 'شو_في',
            'ما في': 'ما_في',
            'يا ريت': 'يا_ريت',
            'يلا بقى': 'يلا_بقى',
            'ما شاء الله': 'ما_شاء_الله',
            'شلون ما': 'شلون_ما',
            'عاد انت': 'عاد_انت',
            'شحال هذا': 'شحال_هذا',
            'باش ندير': 'باش_ندير',
            'ما كاين ش': 'ما_كاين_ش',
            'شنو ماكو': 'شنو_ماكو',
            'اي والله': 'اي_والله',
            'هاي شنو': 'هاي_شنو',
            'ما بعرف ش': 'ما_بعرف_ش',
            'مش عايز': 'مش_عايز',
            'ما في ش': 'ما_في_ش'
        }
        return mwe_dict
    
    def _initialize_morph_features(self):
        """Initialize morphological feature patterns by region."""
        morph_features = {
            'EGY': {
                'present_prefix': ['ب'],
                'future_prefix': ['ح', 'ه'],
                'negation_circum': ['م', 'ش'],
                'progressive': ['بـ', 'عم بـ'],
                'object_suffixes': {
                    '1s': ['ني'],
                    '2ms': ['ك'],
                    '2fs': ['ك'],
                    '3ms': ['ه', 'و'],
                    '3fs': ['ها'],
                    '1p': ['نا'],
                    '2p': ['كو', 'كم'],
                    '3p': ['هم', 'هن']
                }
            },
            'LEV': {
                'present_prefix': ['ب'],
                'future_prefix': ['رح', 'ح'],
                'negation': ['ما', 'مو'],
                'negation_circum': ['م', 'ش'],
                'progressive': ['عم', 'عم بـ'],
                'object_suffixes': {
                    '1s': ['ني'],
                    '2ms': ['ك'],
                    '2fs': ['ك', 'كي'],
                    '3ms': ['ه', 'و'],
                    '3fs': ['ها', 'ا'],
                    '1p': ['نا'],
                    '2p': ['كن', 'كون'],
                    '3p': ['ون', 'هن']
                }
            },
            'GLF': {
                'present_prefix': ['ي', 'أ'],
                'future_prefix': ['ب', 'راح'],
                'negation': ['ما', 'مو', 'مب'],
                'progressive': [''],
                'object_suffixes': {
                    '1s': ['ني'],
                    '2ms': ['ك'],
                    '2fs': ['ج'],
                    '3ms': ['ه'],
                    '3fs': ['ها'],
                    '1p': ['نا'],
                    '2p': ['كم'],
                    '3p': ['هم']
                }
            },
            'MGR': {
                'present_prefix': ['ك', 'ت'],
                'future_prefix': ['غ', 'غادي'],
                'negation_circum': ['ما', 'ش'],
                'progressive': ['كا', 'تا'],
                'object_suffixes': {
                    '1s': ['ني'],
                    '2ms': ['ك'],
                    '2fs': ['ك'],
                    '3ms': ['و', 'ه'],
                    '3fs': ['ها'],
                    '1p': ['نا'],
                    '2p': ['كم'],
                    '3p': ['هم']
                }
            },
            'IRQ': {
                'present_prefix': ['د', 'دا'],
                'future_prefix': ['راح'],
                'negation': ['ما', 'مو'],
                'progressive': ['دا', 'د'],
                'object_suffixes': {
                    '1s': ['ني'],
                    '2ms': ['ك'],
                    '2fs': ['ج'],
                    '3ms': ['ه', 'ة'],
                    '3fs': ['ها'],
                    '1p': ['نا', 'نة'],
                    '2p': ['كم'],
                    '3p': ['هم']
                }
            }
        }
        
        morph_features['SDN'] = {
            'present_prefix': ['ب'],
            'future_prefix': ['ح'],
            'negation': ['ما'],
            'progressive': ['قاعد', 'قاعدة']
        }
        
        morph_features['YEM'] = {
            'present_prefix': ['ي', 'ت', 'ب'],
            'future_prefix': ['با', 'ب'],
            'negation': ['ما', 'مش'],
            'progressive': ['ذي', 'ذا']
        }
        
        return morph_features
    
    def _initialize_city_markers(self):
        """Initialize city-specific lexical markers."""
        city_markers = {
            'Cairo': ['عايز', 'كده', 'دلوقتي', 'فين', 'ازاي', 'بتاع', 'اوضة', 'قوي'],
            'Alexandria': ['عاوز', 'كده', 'دلوقتي', 'فين', 'ازاي'],
            'Aswan': ['عاوز', 'كدا', 'دلوقتي', 'فين', 'قوي'],
            'Beirut': ['بدي', 'هلق', 'هون', 'شو', 'وينك', 'كتير'],
            'Damascus': ['بدي', 'هلأ', 'هون', 'شو', 'كتير'],
            'Aleppo': ['بدي', 'هلأ', 'هون', 'شو', 'كتير'],
            'Jerusalem': ['بدي', 'هيك', 'هون', 'وين', 'كتير'],
            'Amman': ['بدي', 'هيك', 'هون', 'وين', 'كثير'],
            'Salt': ['بدي', 'هيك', 'هون', 'وين', 'كثير'],
            'Doha': ['ابي', 'الحين', 'وينك', 'شلون', 'واجد'],
            'Riyadh': ['ابغى', 'ذا', 'وشلون', 'متى', 'كثير'],
            'Jeddah': ['ابغى', 'دحين', 'فين', 'كيف', 'مرة'],
            'Muscat': ['اريد', 'الحين', 'وين', 'شلون', 'وايد'],
            'Rabat': ['بغيت', 'دابا', 'فين', 'شنو', 'بزاف'],
            'Fes': ['بغيت', 'دابا', 'فين', 'شنو', 'بزاف'],
            'Tripoli': ['نبي', 'حالا', 'وين', 'كيف', 'ياسر', 'هلبة'],
            'Tunis': ['نحب', 'توا', 'وقتاش', 'وين', 'برشا'],
            'Sfax': ['نحب', 'توا', 'وقتاش', 'وين', 'برشا'],
            'Algiers': ['نحب', 'دروك', 'وين', 'كيفاش', 'بزاف'],
            'Benghazi': ['نبي', 'لحين', 'وين', 'كيف', 'بكل'],
            'Baghdad': ['اريد', 'هسة', 'وين', 'شلون', 'هواية'],
            'Basra': ['اريد', 'هسا', 'وين', 'شلون', 'هواية'],
            'Mosul': ['اريد', 'هسع', 'وين', 'شلون', 'كلش'],
            'Khartoum': ['عاوز', 'هسع', 'وين', 'كيف', 'شديد'],
            'Sanaa': ['اشتي', 'دحين', 'فين', 'كيف', 'قوي']
        }
        return city_markers
    
    def _load_madar_lexicon(self, path=None):
        """Load MADAR lexicon from file or use a minimal built-in version."""
        if path and os.path.exists(path):
            try:
                print(f"Loading MADAR lexicon from {path}")
                lexicon = {}
                return lexicon
            except Exception as e:
                print(f"Error loading MADAR lexicon: {e}")
        
        return {
            'room': {
                'Cairo': 'أوضة', 'Alexandria': 'أوضة', 'Aswan': 'أوضة',
                'Beirut': 'غرفة', 'Damascus': 'غرفة', 'Aleppo': 'غرفة',
                'Jerusalem': 'غرفة', 'Amman': 'غرفة', 'Salt': 'غرفة',
                'Tripoli': 'دار', 'Benghazi': 'دار',
                'MSA': 'غرفة'
            },
            'very': {
                'Cairo': 'قوي', 'Alexandria': 'قوي', 'Aswan': 'خالص',
                'Beirut': 'كتير', 'Damascus': 'كتير', 'Aleppo': 'كتير',
                'Jerusalem': 'كتير', 'Amman': 'كثير', 'Salt': 'كثير',
                'Tunis': 'برشا', 'Sfax': 'برشا',
                'Rabat': 'بزاف', 'Fes': 'بزاف', 'Algiers': 'بزاف',
                'Benghazi': 'بكل', 'Tripoli': 'هلبة',
                'Baghdad': 'هواية', 'Basra': 'هواية', 'Mosul': 'كلش',
                'Doha': 'واجد', 'Riyadh': 'جدا', 'Jeddah': 'مرة', 'Muscat': 'وايد',
                'Khartoum': 'شديد',
                'Sanaa': 'قوي',
                'MSA': 'جداً'
            },
            'what': {
                'Cairo': 'إيه', 'Alexandria': 'إيه', 'Aswan': 'إيه',
                'Beirut': 'شو', 'Damascus': 'شو', 'Aleppo': 'شنو',
                'Jerusalem': 'شو', 'Amman': 'شو', 'Salt': 'شو',
                'Doha': 'شنو', 'Riyadh': 'وش', 'Jeddah': 'إيش', 'Muscat': 'شو',
                'Rabat': 'شنو', 'Fes': 'شنو', 'Tripoli': 'شن',
                'Tunis': 'شنوة', 'Sfax': 'شنية', 'Algiers': 'واش',
                'Benghazi': 'شن',
                'Baghdad': 'شنو', 'Basra': 'شنو', 'Mosul': 'شنو',
                'Khartoum': 'شنو',
                'Sanaa': 'ايش',
                'MSA': 'ماذا'
            },
            'now': {
                'Cairo': 'دلوقتي', 'Alexandria': 'دلوقتي', 'Aswan': 'دلوقتي',
                'Beirut': 'هلق', 'Damascus': 'هلأ', 'Aleppo': 'هلأ',
                'Jerusalem': 'هيك', 'Amman': 'هسا', 'Salt': 'هلا',
                'Doha': 'الحين', 'Riyadh': 'الحين', 'Jeddah': 'دحين', 'Muscat': 'الحين',
                'Rabat': 'دابا', 'Fes': 'دابا', 'Tripoli': 'توا',
                'Tunis': 'توا', 'Sfax': 'توا', 'Algiers': 'دروك',
                'Benghazi': 'تو',
                'Baghdad': 'هسة', 'Basra': 'هسة', 'Mosul': 'هسع',
                'Khartoum': 'هسع',
                'Sanaa': 'ذحين',
                'MSA': 'الآن'
            },
            'want': {
                'Cairo': 'عايز', 'Alexandria': 'عاوز', 'Aswan': 'عاوز',
                'Beirut': 'بدي', 'Damascus': 'بدي', 'Aleppo': 'بدي',
                'Jerusalem': 'بدي', 'Amman': 'بدي', 'Salt': 'بدي',
                'Doha': 'أبي', 'Riyadh': 'أبغى', 'Jeddah': 'أبغى', 'Muscat': 'أبا',
                'Rabat': 'بغيت', 'Fes': 'بغيت', 'Tripoli': 'نبي',
                'Tunis': 'نحب', 'Sfax': 'نحب', 'Algiers': 'نحب',
                'Benghazi': 'نبي',
                'Baghdad': 'أريد', 'Basra': 'أريد', 'Mosul': 'أريد',
                'Khartoum': 'داير',
                'Sanaa': 'أشتي',
                'MSA': 'أريد'
            }
        }
    
    def normalize_chars(self, text):
        """Normalize Arabic characters (e.g., alef variants, taa marbuta)."""
        if not text:
            return ""
        
        for source, target in self.char_map.items():
            text = text.replace(source, target)
        
        text = re.sub(r'اا+', 'ا', text)
        text = re.sub(r'وو+', 'و', text)
        text = re.sub(r'يي+', 'ي', text)
        
        return text
    
    def clean_text(self, text):
        """Remove diacritics, punctuation, tatweel, and extra spaces."""
        text = strip_tashkeel(text)
        text = strip_diacritics(text)
        text = strip_tatweel(text)
        
        for char in self.punctuations_list:
            if char != '_':
                text = text.replace(char, ' ')
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def detect_mwes(self, text):
        """Replace known multi-word expressions with single tokens."""
        for mwe, replacement in self.mwe_dict.items():
            text = text.replace(mwe, replacement)
        return text
    
    def apply_dialect_phonology(self, text, city):
        """Apply city-specific phonological changes based on CAPHI map."""
        if not city or city not in self.city_to_region:
            return text
        
        if city not in self.cities:
            region = self.city_to_region.get(city)
            if not region:
                return text
            for c, r in self.city_to_region.items():
                if r == region:
                    city = c
                    break
        
        words = text.split()
        processed_words = []
        
        for word in words:
            processed_word = word
            for char, city_map in self.caphi_map.items():
                if char in processed_word and city in city_map:
                    processed_word = processed_word.replace(char, city_map[city])
            
            processed_words.append(processed_word)
            
        return ' '.join(processed_words)
    
    def apply_morphological_features(self, text, city):
        """Apply morphological features (prefixes, clitics) for a city/region."""
        if not city or city not in self.city_to_region:
            return text
            
        region = self.city_to_region[city]
        if region not in self.morph_features:
            return text
            
        words = text.split()
        processed_words = []
        
        features = self.morph_features[region]
        i = 0
        while i < len(words):
            word = words[i]
            
            if region in ['EGY', 'LEV', 'MGR'] and i < len(words) - 1:
                if (word.startswith('م') or word == 'ما' or word == 'مش') and words[i+1].endswith('ش'):
                    processed_words.append(f"{word}_{words[i+1]}")
                    i += 2
                    continue
            
            prefix_detected = False
            for prefix_type in ['present_prefix', 'future_prefix', 'progressive']:
                if prefix_type in features:
                    for prefix in features[prefix_type]:
                        if len(prefix) > 0 and word.startswith(prefix) and len(word) > len(prefix) + 2:
                            word = f"{prefix}_{word[len(prefix):]}"
                            prefix_detected = True
                            break
                    if prefix_detected:
                        break
                        
            processed_words.append(word)
            i += 1
            
        return ' '.join(processed_words)
    
    def normalize_lexical_entries(self, text, city):
        """Normalize words to the city's variant using the MADAR lexicon."""
        if not city or not self.madar_lexicon:
            return text
            
        words = text.split()
        normalized = []
        
        for word in words:
            normalized_word = word
            for concept, city_variants in self.madar_lexicon.items():
                for dialect_city, dialect_word in city_variants.items():
                    if word == dialect_word and dialect_city != city:
                        if city in city_variants:
                            normalized_word = city_variants[city]
                        elif 'MSA' in city_variants:
                            normalized_word = city_variants['MSA']
                        break
            
            normalized.append(normalized_word)
        
        return ' '.join(normalized)
    
    def extract_dialectal_features(self, text, city=None):
        """Extract dialect-specific features (markers, phonology, morphology)."""
        features = {}
        
        if city and city in self.city_markers:
            words = set(text.split())
            city_words = set(self.city_markers[city])
            features['has_city_markers'] = len(words.intersection(city_words)) > 0
            features['city_marker_ratio'] = len(words.intersection(city_words)) / len(words) if words else 0
        
        if city:
            region = self.city_to_region.get(city, '')
            
            for char, city_map in self.caphi_map.items():
                if city in city_map:
                    dialectal_sound = city_map[city]
                    standard_sound = char
                    features[f'has_{char}_as_{dialectal_sound}'] = char in text
            
            if region in self.morph_features:
                morph = self.morph_features[region]
                
                if 'present_prefix' in morph:
                    features['has_present_prefix'] = any(
                        any(word.startswith(prefix) for prefix in morph['present_prefix'])
                        for word in text.split()
                    )
                
                if 'future_prefix' in morph:
                    features['has_future_prefix'] = any(
                        any(word.startswith(prefix) for prefix in morph['future_prefix'])
                        for word in text.split()
                    )
                
                if 'negation' in morph:
                    features['has_negation'] = any(
                        word in morph['negation']
                        for word in text.split()
                    )
                
                if 'negation_circum' in morph and len(morph['negation_circum']) >= 2:
                    prefix, suffix = morph['negation_circum'][0], morph['negation_circum'][1]
                    features['has_negation_circum'] = any(
                        word.startswith(prefix) and word.endswith(suffix)
                        for word in text.split()
                    )
        
        return features
    
    def preprocess(self, text, city=None, embedding_model='ARABERT'):
        """Full preprocessing pipeline for Arabic dialect text."""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Basic cleaning and normalization for all text
        text = self.normalize_chars(text)
        text = self.clean_text(text)
        text = self.detect_mwes(text)
        
        # Define dialects with more data available
        high_resource_dialects = ['Cairo', 'Tunis', 'Rabat', 'Beirut', 'Doha', 'MSA']
        
        # Skip dialect-specific steps for MSA
        if city == 'MSA':
            pass 
            
        # Apply full processing for high-resource dialects
        elif city in high_resource_dialects:
            text = self.apply_dialect_phonology(text, city)
            text = self.apply_morphological_features(text, city)
            text = self.normalize_lexical_entries(text, city)
            
        # For low-resource dialects, apply phonology and borrow features
        elif city:
            text = self.apply_dialect_phonology(text, city)
            
            # Map low-resource cities to higher-resource ones in the same country
            same_country_mapping = {
                'Alexandria': 'Cairo',
                'Aswan': 'Cairo',
                'Damascus': 'Beirut',
                'Aleppo': 'Beirut',
                'Jerusalem': 'Beirut',
                'Amman': 'Beirut',
                'Salt': 'Beirut',
                'Jeddah': 'Riyadh',
                'Fes': 'Rabat',
                'Sfax': 'Tunis',
                'Benghazi': 'Tripoli',
                'Basra': 'Baghdad',
                'Mosul': 'Baghdad'
            }
            
            # Define representative cities for each region
            region_representatives = {
                'EGY': 'Cairo', 
                'LEV': 'Beirut', 
                'GLF': 'Doha',
                'MGR': 'Tunis', 
                'IRQ': 'Baghdad', 
                'SDN': 'Khartoum', 
                'YEM': 'Sanaa'
            }
            
            # Find a suitable "donor" city for morphological/lexical features
            donor_city = None
            if city in same_country_mapping:
                donor_city = same_country_mapping[city]
            
            # If no same-country donor or donor is also low-resource, use region representative
            if not donor_city or donor_city not in high_resource_dialects:
                region = self.city_to_region.get(city, '')
                if region in region_representatives:
                    donor_city = region_representatives[region]
                else:
                    donor_city = city # Fallback to original city if region unknown
            
            # Apply features from the donor city
            text = self.apply_morphological_features(text, donor_city)
            text = self.normalize_lexical_entries(text, city) # Lexical still uses original city
        
        return text.strip()

# Initialize preprocessor instance for potential use elsewhere
# madar_preprocessor = MADARPreprocessor()
