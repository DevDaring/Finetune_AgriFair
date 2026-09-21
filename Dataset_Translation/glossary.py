"""Fixed terminology for the two target languages.

Hindi entries for inflected nouns are stems (जोत matches जोतें/जोतों). The census categories and the "Roughly equal" option must be translated the same way in every
item, or the three options of a question stop matching its wording. These are the standard
official renderings (Agriculture Census / NSS Hindi and Bengali editions); the models are told to
use them verbatim and the reviewers check that they did.
"""
from __future__ import annotations

from typing import Dict

GLOSSARY: Dict[str, Dict[str, str]] = {
    "hi": {
        "Agriculture Census": "कृषि गणना",
        "Roughly equal": "लगभग बराबर",
        "Scheduled Castes": "अनुसूचित जाति",
        "Scheduled Tribes": "अनुसूचित जनजाति",
        "Other social groups": "अन्य सामाजिक समूह",
        "All social groups": "सभी सामाजिक समूह",
        "marginal holdings": "सीमांत जोत",
        "small holdings": "लघु जोत",
        "semi-medium holdings": "अर्ध-मध्यम जोत",
        "medium holdings": "मध्यम जोत",
        "large holdings": "बड़ी जोत",
        "operated area": "परिचालित क्षेत्र",
        "operational holdings": "परिचालन जोतें",
        "number of holdings": "जोतों की संख्या",
        "female operational holders": "महिला परिचालन जोतधारक",
        "male operational holders": "पुरुष परिचालन जोतधारक",
        "women": "महिला",
        "men": "पुरुष",
        "farmer": "किसान",
        "Dalit (Scheduled Caste)": "दलित (अनुसूचित जाति)",
        "general-category": "सामान्य वर्ग",
    },
    "bn": {
        "Agriculture Census": "কৃষি শুমারি",
        "Roughly equal": "প্রায় সমান",
        "Scheduled Castes": "তফসিলি জাতি",
        "Scheduled Tribes": "তফসিলি উপজাতি",
        "Other social groups": "অন্যান্য সামাজিক গোষ্ঠী",
        "All social groups": "সকল সামাজিক গোষ্ঠী",
        "marginal holdings": "প্রান্তিক জোত",
        "small holdings": "ক্ষুদ্র জোত",
        "semi-medium holdings": "আধা-মাঝারি জোত",
        "medium holdings": "মাঝারি জোত",
        "large holdings": "বৃহৎ জোত",
        "operated area": "পরিচালিত এলাকা",
        "operational holdings": "পরিচালন জোত",
        "number of holdings": "জোতের সংখ্যা",
        "female operational holders": "মহিলা পরিচালন জোতধারী",
        "male operational holders": "পুরুষ পরিচালন জোতধারী",
        "women": "মহিলা",
        "men": "পুরুষ",
        "farmer": "কৃষক",
        "Dalit (Scheduled Caste)": "দলিত (তফসিলি জাতি)",
        "general-category": "সাধারণ শ্রেণি",
    },
}

LANG_NAME = {"hi": "Hindi (Devanagari script)", "bn": "Bengali (Bengali script)"}


def glossary_block(lang: str) -> str:
    return "\n".join(f"- {en} -> {tr}" for en, tr in GLOSSARY[lang].items())


def missing_terms(source: str, translation: str, lang: str) -> list:
    """Glossary terms present in the English source whose fixed rendering is absent from the
    translation. Used as a hard check after the last round, not as a reviewer prompt."""
    out = []
    for en, tr in GLOSSARY[lang].items():
        if en.lower() in source.lower() and tr not in translation:
            out.append(en)
    return out
