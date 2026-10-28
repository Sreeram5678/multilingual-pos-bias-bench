"""
Multilingual Needle-In-A-Haystack (M-NIAH) Data Generator
Generates controlled, paired long-context retrieval instances across English, Hindi, and Telugu.
Guarantees strict experimental control by holding distractors constant while shifting only the needle depth.
"""

import random
import uuid
from typing import List, Dict, Any, Tuple

# Competitive distractor project names across languages
PROJECT_NAMES = {
    "en": ["Apollo", "Pegasus", "Atlas", "Hermes", "Titan", "Vanguard", "Prometheus", "Hyperion", "Phoenix", "Centaur"],
    "hi": ["अपोलो", "पेगासस", "एटलस", "हर्मिस", "टाइटन", "वानगार्ड", "प्रोमेथियस", "हाइपरियन", "फीनिक्स", "सेंटौर"],
    "te": ["అపోలో", "పెగాసస్", "అట్లాస్", "హెర్మిస్", "టైటాన్", "వాన్‌గార్డ్", "ప్రోమోథియస్", "హైపెరియన్", "ఫీనిక్స్", "సెంటార్"]
}

# Competitive distractor fact templates with matching syntactic structure to the needle
DISTRACTOR_TEMPLATES = {
    "en": [
        "SYSTEM LOG: The designated access passkey for Project {project} is {code}.",
        "SECURITY NOTICE: The operational verification key for Project {project} was logged as {code}.",
        "CLEARANCE RECORD: Project {project} issued authorization credential {code} to engineering.",
        "TELEMETRY ARCHIVE: The security dispatch token for Project {project} is registered as {code}.",
        "ACCESS AUDIT: Verification checkpoint recorded passkey {code} assigned to Project {project}."
    ],
    "hi": [
        "सिस्टम लॉग: प्रोजेक्ट {project} के लिए निर्धारित एक्सेस पासकी {code} है।",
        "सुरक्षा सूचना: प्रोजेक्ट {project} के लिए सत्यापन कुंजी {code} के रूप में दर्ज की गई थी।",
        "निकासी रिकॉर्ड: प्रोजेक्ट {project} ने इंजीनियरिंग को प्राधिकरण क्रेडेंशियल {code} जारी किया।",
        "टेलीमेट्री संग्रह: प्रोजेक्ट {project} के लिए सुरक्षा डिस्पैच टोकन {code} के रूप में पंजीकृत है।",
        "एक्सेस ऑडिट: सत्यापन चौकी ने प्रोजेक्ट {project} को सौंपी गई पासकी {code} दर्ज की।"
    ],
    "te": [
        "సిస్టమ్ లాగ్: ప్రాజెక్ట్ {project} కోసం కేటాయించిన యాక్సెస్ పాస్‌కీ {code}.",
        "భద్రతా నోటీసు: ప్రాజెక్ట్ {project} కోసం కార్యాచరణ ధృవీకరణ కీ {code}గా లాగ్ చేయబడింది.",
        "క్లియరెన్స్ రికార్డ్: ప్రాజెక్ట్ {project} ఇంజనీరింగ్‌కు అధికార ఆధారాలు {code} జారీ చేసింది.",
        "టెలిమెట్రీ ఆర్కైవ్: ప్రాజెక్ట్ {project} కోసం సెక్యూరిటీ డిస్పాచ్ టోకెన్ {code}గా నమోదు చేయబడింది.",
        "యాక్సెస్ ఆడిట్: వెరిఫికేషన్ చెక్‌పాయింట్ ప్రాజెక్ట్ {project}కు కేటాయించిన పాస్‌కీ {code}ని నమోదు చేసింది."
    ]
}

# The target "Needle" templates and Questions
NEEDLE_TEMPLATES = {
    "en": {
        "statement": "IMPORTANT NOTE: The designated master passkey for Project Orion is {gold_key}.",
        "question": "What is the designated master passkey for Project Orion? Provide ONLY the exact passkey and nothing else.",
        "header": "Below is a collection of operational system logs:\n\n",
        "footer": "\n\nBased exclusively on the logs provided above, answer the following question:\nQuestion: "
    },
    "hi": {
        "statement": "महत्वपूर्ण सूचना: प्रोजेक्ट ओरियन के लिए निर्धारित मास्टर पासकी {gold_key} है।",
        "question": "प्रोजेक्ट ओरियन के लिए निर्धारित मास्टर पासकी क्या है? केवल सटीक पासकी लिखें, कोई अन्य शब्द नहीं।",
        "header": "नीचे परिचालन प्रणाली लॉग का एक संग्रह दिया गया है:\n\n",
        "footer": "\n\nऊपर दिए गए लॉग के आधार पर, निम्नलिखित प्रश्न का उत्तर दें:\nप्रश्न: "
    },
    "te": {
        "statement": "ముఖ్యమైన గమనిక: ప్రాజెక్ట్ ఓరియన్ కోసం కేటాయించిన మాస్టర్ పాస్‌కీ {gold_key}.",
        "question": "ప్రాజెక్ట్ ఓరియన్ కోసం కేటాయించిన మాస్టర్ పాస్‌కీ ఏమిటి? కేవలం ఖచ్చితమైన పాస్‌కీని మాత్రమే సమాధానంగా ఇవ్వండి.",
        "header": "క్రింద సిస్టమ్ కార్యాచరణ లాగ్‌ల సమాహారం ఇవ్వబడింది:\n\n",
        "footer": "\n\nపైన అందించిన లాగ్‌ల ఆధారంగా మాత్రమే, ఈ క్రింది ప్రశ్నకు సమాధానం ఇవ్వండి:\nప్రశ్న: "
    }
}

def generate_random_key(rng: random.Random) -> str:
    """Generates a deterministic hexadecimal key using the provided seeded RNG, e.g. 'A7F3-9D2B'"""
    chars = "0123456789ABCDEF"
    u = "".join(rng.choices(chars, k=8))
    return f"{u[:4]}-{u[4:8]}"

def generate_haystack_items(lang: str, target_count: int, rng: random.Random) -> Tuple[List[str], List[str]]:
    """
    Generates a list of competitive distractor sentences and the list of distractor keys.
    """
    templates = DISTRACTOR_TEMPLATES[lang]
    project_pool = PROJECT_NAMES[lang]
    distractor_sentences = []
    distractor_keys = []

    for i in range(target_count):
        tmpl = templates[i % len(templates)]
        proj = project_pool[i % len(project_pool)]
        distractor_key = generate_random_key(rng)
        distractor_keys.append(distractor_key)
        distractor_sentences.append(tmpl.format(project=proj, code=distractor_key))

    return distractor_sentences, distractor_keys

def build_instance(
    lang: str,
    depth: float,
    num_distractors: int,
    gold_key: str,
    distractor_sentences: List[str]
) -> Dict[str, Any]:
    """
    Constructs a prompt instance by placing the gold needle at exact relative depth.
    depth: float between 0.0 (very start) and 1.0 (very end).
    """
    # Create a copy so distractor ordering remains strictly constant
    haystack = list(distractor_sentences)
    insert_idx = int(round(depth * len(haystack)))
    insert_idx = min(max(0, insert_idx), len(haystack))

    needle_stmt = NEEDLE_TEMPLATES[lang]["statement"].format(gold_key=gold_key)
    haystack.insert(insert_idx, needle_stmt)

    context_body = "\n".join(haystack)
    prompt = (
        NEEDLE_TEMPLATES[lang]["header"] +
        context_body +
        NEEDLE_TEMPLATES[lang]["footer"] +
        NEEDLE_TEMPLATES[lang]["question"]
    )

    return {
        "prompt": prompt,
        "gold_key": gold_key,
        "depth": depth,
        "insert_idx": insert_idx,
        "total_items": len(haystack),
        "language": lang,
        "haystack_chunks": haystack
    }

def generate_paired_dataset(
    languages: List[str],
    depth_bins: List[float],
    samples_per_cell: int,
    num_distractors: int = 40,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Generates a paired evaluation benchmark.
    For each base instance, distractors and gold key are created ONCE,
    then tested at EVERY depth position.
    """
    rng = random.Random(seed)
    dataset = []

    for lang in languages:
        for sample_i in range(samples_per_cell):
            gold_key = generate_random_key(rng)
            distractors, distractor_keys = generate_haystack_items(lang, num_distractors, rng)

            for depth in depth_bins:
                inst = build_instance(
                    lang=lang,
                    depth=depth,
                    num_distractors=num_distractors,
                    gold_key=gold_key,
                    distractor_sentences=distractors
                )
                inst["sample_id"] = f"{lang}_s{sample_i:03d}"
                inst["distractor_keys"] = distractor_keys
                dataset.append(inst)

    return dataset

if __name__ == "__main__":
    test_data = generate_paired_dataset(
        languages=["en", "hi", "te"],
        depth_bins=[0.0, 0.5, 1.0],
        samples_per_cell=2,
        num_distractors=10
    )
    print(f"Generated {len(test_data)} test instances.")
    sample = test_data[0]
    print(f"Sample Language: {sample['language']}, Depth: {sample['depth']}")
    print(f"Gold Key: {sample['gold_key']}")
    print("Prompt Preview (first 250 chars):")
    print(sample['prompt'][:250] + "...")
