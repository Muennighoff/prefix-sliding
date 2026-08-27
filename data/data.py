from functools import partial
import json
import random

import datasets

from decontaminate_util import *

# Use smaller writer batch size, e.g. 200 for large datasets to avoid OOM. Default to 1000.
# Large datasets (>1GB): LiveCodeBench, MATH, USACO
LARGE_DATASET_WRITER_BATCH_SIZE = 1000

BAD_OMNIMATH_SAMPLES = [
    {"question": "Let $\\mathbb{R}$ be the set of real numbers .  Determine all functions $f\u00a0: \\mathbb{R} \\rightarrow \\mathbb{R}$ such that\n  \nfor all pairs of real numbers $x$ and $y$ ."},
    {"question": "Find the sum of the ages of everyone who wrote a problem for this year's HMMT November contest. If your answer is $X$ and the actual value is $Y$, your score will be $\\max (0,20-|X-Y|)$"},
]

# All datasets have the same columns
DS_COLUMNS = ["problem", "solution", "tests", "domain", "source", "metadata"]
# "domain": 'astronomy', 'biology', 'chemistry', 'code', 'crossword', 'math', 'medical', 'physics'

### Load functions ###
def load_omnimath():
    ds = datasets.load_dataset("KbsdJames/Omni-MATH", trust_remote_code=True, split="test")
    ds = ds.filter(lambda x: x["problem"] not in BAD_OMNIMATH_SAMPLES)
    ds = ds.filter(lambda x: x["difficulty"] >= 3.5) # Only keep hard questions
    ds = ds.map(lambda x: {"problem": x.pop("problem"), "solution": x.pop("solution"), "domain": "math", "source": "KbsdJames/Omni-MATH", "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_math():
    ds = datasets.load_dataset("simplescaling/openaimath", trust_remote_code=True)["train"]
    ds = ds.filter(lambda x: x["level"] > 3) # Remove geometry as it's not well-suited for RL
    ds = ds.map(lambda x: {"problem": x.pop("problem"), "solution": x.pop("answer"), "domain": "math", "source": "simplescaling/openaimath/" + x['subject'], "metadata": str(x)},
                writer_batch_size=LARGE_DATASET_WRITER_BATCH_SIZE)
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_numinamath():
    ds = datasets.load_dataset("AI-MO/NuminaMath-1.5", trust_remote_code=True)["train"]

    # Remove incomplete problems
    ds = ds.filter(lambda x: "Yes" in x["problem_is_valid"])
    # Remove incomplete problems
    ds = ds.filter(lambda x: "Yes" in x["solution_is_valid"])
    # Buggy questions
    ds = ds.filter(lambda x: x["answer"] is not None)
    ds = ds.filter(lambda x: "notfound" not in x["answer"])
    ds = ds.filter(lambda x: "not found" not in x["answer"])    

    ### RL-specific filters
    # Remove proofs to avoid subjectivity in grading
    ds = ds.filter(lambda x: ("proof" not in x["question_type"]) and ("proof" not in x["answer"]))
    ds = ds.filter(lambda x: ("MCQ" not in x["question_type"]))
    # Some questions inadvertently don't have proof in answer but are proofs
    ds = ds.filter(lambda x: "prove that" not in x["problem"].lower())

    ds = ds.map(lambda x: {"problem": x.pop("problem"), "solution": x.pop("answer"), "domain": "math", "source": "AI-MO/NuminaMath-1.5/" + x["source"], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_olympic_arena():
    # TODO: Fix 2 coding samples that have tests in metadata from GAIR
    confs = ['Math', 'Physics', 'Chemistry', 'Biology', 'Geography', 'Astronomy', 'CS']
    subject_to_o1domain = {"CS": "code"}
    
    # ds = [datasets.load_dataset("GAIR/OlympicArena", c, trust_remote_code=True) for c in confs]
    # ds = datasets.concatenate_datasets([d['test'] for d in ds] + [d['val'] for d in ds])

    ### RL-specific
    ds = datasets.concatenate_datasets([datasets.load_dataset("GAIR/OlympicArena", c, split="val", trust_remote_code=True) for c in confs])
    # https://github.com/GAIR-NLP/OlympicArena/blob/99f6fa745ff8429f2951fd56a0d0ec580a9e00e2/annotation/main.py#L18
    # Could consider allowing "MC" as if multiple correct with 4 options, chances of correct guess are just 6.25%
    ds = ds.filter(lambda x: x["answer_type"] in ["NV", "SET", "IN", "EX", "EQ", "TUP", "MPV", "MA"])

    # Filter for EN & text-only
    ds = ds.filter(lambda x: (x["language"] == "EN") and (x["modality"] == "text-only"))
    ds = ds.map(lambda x: {"problem": x.pop("problem"), "solution": x.pop("solution"), "domain": subject_to_o1domain.get(x['subject'], x['subject'].lower()), "source": "GAIR/OlympicArena/" + x['subject'], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_theoremqa():
    ds = datasets.load_dataset("TIGER-Lab/TheoremQA", trust_remote_code=True)["test"]
    ds = ds.filter(lambda x: x["Picture"] is None)

    ### RL-specific
    ds = ds.filter(lambda x: x["Answer_type"] not in ["bool", "option"])

    ds = ds.map(lambda x: {"problem": x.pop("Question"), "solution": x.pop("Answer"), "domain": "math", "source": "TIGER-Lab/TheoremQA/" + x['Answer_type'], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_scieval():
    """
    Category Physics; Task: SocraticQA
    What is the moment of inertia of a pendulum with a mass of $2 kg$ that is $7  m$ from the pivot?\n\nA. 56 kgm^2\nB. 196 kgm^2\nC. 84 kgm^2\nD. 98 kgm^2\n\nAnswer:

    Category Chemistry; Task: SocraticQA
    What is the molecular geometry of the $PF_3$ molecule?\n\nA. Trigonal planar\nB. Bent\nC. Trigonal pyramidal\nD. Tetrahedral\n\nAnswer:
    Category Chemistry; Task: reagent selection
    Given the rest of reaction components:\nreactant 1: Ic1ccc2ncccc2c1\nreactant 2: Cc1ccc2c(cnn2C2CCCCO2)c1B1OC(C)(C)C(C)(C)O1\nligand: c1ccc(P(c2ccccc2)c2ccccc2)cc1\nbase: C(=O)(O)[O-].[Na+]  \nSolvent list for selection:\nC1CCOC1,CN(C)C=O,CO\nOptimal solvent:

    Category Biology; Task: MedQA
    A 74-year-old man was admitted to the intensive care ward due to progressive dyspnea, cough with pink sputum, and diaphoresis. He had 2 myocardial infarctions at the age of 66 and 69 years and suffers from chronic heart failure. At the time of presentation, his vital signs are as follows: blood pressure 90/50 mm Hg, heart rate 108/min, respiratory rate 29/min, and temperature 35.5°C (95.9°F). On physical examination, the patient sits upright. He is lethargic and cyanotic. Lung auscultation reveals widespread bilateral fine rales. Cardiac examination is significant for S3, accentuation of the pulmonic component of S2, and a systolic murmur heard best at the apex of the heart. Soon after hospitalization, the patient develops ventricular fibrillation and dies despite adequate resuscitation measures. Which microscopic finding would you expect to see in this patient on autopsy?\n\nA. Brownish inclusions in the pulmonary macrophages on H&E staining\nB. Positive Prussian-blue staining of the kidney tissue\nC. Ground-glass hepatocytes\nD. Positive Congo-red staining of the cardiac tissue\n\nAnswer:
    Category Biology; Task: PubMedQA
    Polymorphisms in the oestrogen receptor 1 (ESR1) and oestrogen receptor 2 (ESR2) genes are associated with intermediate or endpoint markers of cardiovascular disease and with the efficacy of postmenopausal hormone therapy (HT). Contradictory findings have been described in the past and the role of these genetics variants remains unclear.\nA cross-sectional study was carried out with 266 postmenopausal women, of whom 115 received oral HT (HT+) and 151 did not receive any HT (HT-). We analysed three single-nucleotide polymorphisms (SNPs) in ESR1 (rs1801132, rs7757956 and rs2813544) and two in ESR2 (rs3020450 and rs7154455) and derived haplotypes with three additional polymorphisms that had been previously investigated by our group (ESR1 rs2234693 and ESR2 rs1256049 and rs4986938).\nThe ESR1 rs2813544 polymorphism was associated with low-density lipoprotein cholesterol (LDL-C) in HT+ postmenopausal women (p\u2009=\u20090.044; pC\u2009=\u20090.388), while one ESR2 gene haplotype was associated with total cholesterol (T-chol) (p\u2009=\u20090.015; pC\u2009=\u20090.090) and LDL-C in HT+ postmenopausal women (p\u2009=\u20090.021; pC\u2009=\u20090.126).\n\nAre polymorphisms in oestrogen receptors genes associated with lipid levels in response to hormone therapy?\n\nAnswer:
    Category Biology; Task: SocraticQA
    What substance is transported across the inner membrane of the mitochondria?\n\nA. Glucose\nB. Protons\nC. Oxygen\nD. Electrons\n\nAnswer:
    """
    pass # Not adequate for RL as all questions are MCQs

def load_olympiad_bench():
    # Only EN & TO (text-only); Both OE (open-ended) and TP (Theorem proof)
    confs = ["OE_TO_maths_en_COMP", "OE_TO_physics_en_COMP", "TP_TO_maths_en_COMP", "TP_TO_physics_en_COMP"]

    ### RL-specific 
    confs = ["OE_TO_maths_en_COMP", "OE_TO_physics_en_COMP"] # No proofs

    # Multimodal: "OE_MM_maths_en_COMP", "OE_MM_physics_en_COMP", "TP_MM_maths_en_COMP", "TP_MM_physics_en_COMP"
    ds = [datasets.load_dataset("Hothan/OlympiadBench", c, trust_remote_code=True)['train'] for c in confs]
    ds = datasets.concatenate_datasets(ds)

    def add_context(x):
        if x['context'] is not None:
            x['question'] = "Context:\n\n" + x['context'] + "\n\nQuestion to answer:\n\n" + x['question']
        return x
    ds = ds.map(add_context)

    # The physics one is also rather math-heavy; 4 samples with two correct answers
    ds = ds.map(lambda x: {"problem": x.pop("question"), "solution": " or ".join(x.pop("final_answer")), "domain": x.get("subject").lower(), "source": "Hothan/OlympiadBench/" + x['question_type'] + "/" + x.pop('subject'), "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_jeebench():
    ds = datasets.load_dataset("daman1209arora/jeebench", trust_remote_code=True)['test']

    ### RL-specific
    ds = ds.filter(lambda x: x["type"] in ["Integer", "Numeric"])

    subject_to_o1domain = {"math": "math", "phy": "physics", "chem": "chemistry"}
    ds = ds.map(lambda x: {"problem": x.pop("question"), "solution": x.pop("gold"), "domain": subject_to_o1domain[x['subject']], "source": "daman1209arora/jeebench/" + x['subject'], "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_agieval():
    ### RL-specifc all questions are MCQs except math_agieval
    # Only take most difficult questions; 'options' field not present
    ds = datasets.load_dataset("baber/agieval", 'math_agieval', trust_remote_code=True).filter(lambda x: x['level'] == 5).map(lambda x: {"problem": x.pop("question"), "solution": x.pop("solution"), "domain": "math", "source": "baber/agieval/math_agieval", "metadata": str(x)})
    ds = datasets.concatenate_datasets([ds['test'], ds['few_shot']])
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_s1prob():
    pass # Not adequate for RL as all questions are proofs

def load_gpqa_extended():
    ds = datasets.load_dataset("Idavidrein/gpqa", "gpqa_extended", trust_remote_code=True)['train']
    # Filter against diamond
    ds_diamond = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", trust_remote_code=True)['train']
    ds = ds.filter(lambda x: x["Question"] not in ds_diamond["Question"])

    ### RL-specific
    # Filter out Questions likely relying on the options
    ds = ds.filter(lambda x: "which" not in x["Question"].lower())
    ds = ds.filter(lambda x: "options" not in x["Question"].lower())
    ds = ds.filter(lambda x: "following" not in x["Question"].lower())

    ds = ds.map(lambda x: {"problem": x.pop("Question"), "solution": x.pop("Correct Answer"), "domain": x.pop('High-level domain').lower(), "source": "Idavidrein/gpqa", "metadata": str(x)})

    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_xword():
    ds = datasets.load_dataset("0xharib/xword1", trust_remote_code=True)['train']

    # Use slightly different format, e.g. would need to Rename instruction, input, output -> cl
    # ds2 = datasets.load_dataset("0xharib/xword2", trust_remote_code=True)['train']
    # ds3 = datasets.load_dataset("0xharib/xword3", trust_remote_code=True)['train']
    # ds = datasets.concatenate_datasets([ds1, ds2, ds3])

    instruction = "Solve the crossword puzzle given the clue and number of letters in brackets."
    ds = ds.map(lambda x: {"problem": instruction + "\n\n" + x.pop("input").split("### Clue: ")[1], "solution": x["output"].split("### Answer:")[1].split("### Explanation:")[0].strip(), "domain": "crossword", "source": "0xharib/xword1", "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_usaco():
    pass # TODO
    ds = datasets.load_dataset("codegenning/usacobench_formatted")['test']
    ds = ds.map(lambda x: {"problem": x.pop("question").strip(), "solution": None, "domain": "code", "source": "codegenning/usacobench_formatted", "metadata": str(x)},
                writer_batch_size=LARGE_DATASET_WRITER_BATCH_SIZE)
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_s1teasers():
    ds = datasets.load_dataset("simplescaling/s1-teasers")['train']
    ds = ds.map(lambda x: {"problem": x.pop("Question").strip(), "solution": x.pop("Answer"), "domain": "math", "source": "simplescaling/s1-teasers", "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_livecodebench():
    pass # TODO
    versions = ["release_v1", "release_v2", "release_v3"]
    datasets_list = []
    for version in versions:
        ds = datasets.load_dataset("livecodebench/code_generation_lite", version_tag=version, trust_remote_code=True)["test"]
        ds = ds.map(lambda x: {
                "problem": x.pop("question_content").strip(),
                "solution": None,
                "domain": "code",
                "source": f"LiveCodeBench/{version}",
                "metadata": str(x)
            }, writer_batch_size=LARGE_DATASET_WRITER_BATCH_SIZE)
        # filter only the difficult questions
        ds = ds.filter(lambda x: x["difficulty"] == "hard")
        ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
        datasets_list.append(ds)
      
    final_ds = datasets.concatenate_datasets(datasets_list)
    return final_ds

def load_aime():
    ds = datasets.load_dataset("di-zhang-fdu/AIME_1983_2024")['train']
    ds = ds.filter(lambda x: int(x['ID'].split('-')[0]) < 2024) # Leave it for eval
    ds = ds.map(lambda x: {"problem": x.pop("Question").strip(), "solution": x.pop("Answer"), "domain": "math", "source": "qq8933/AIME_1983_2024", "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_hle():
    ds = datasets.load_dataset("cais/hle")['test']
    ds = ds.filter(lambda x: (x['answer_type'] == 'exactMatch') and not(x['image']))
    ds = ds.map(lambda x: {"problem": x.pop("question").strip(), "solution": x.pop("answer"), "domain": "math", "source": "cais/hle", "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_linguini():
    from ast import literal_eval
    ds = datasets.load_dataset("facebook/linguini")['test']
    def format_answer(x):
        a = literal_eval(x['answer'])
        if x['eval_type'] == 'multi':
            a = "\n".join([" or ".join(l) for l in a])
        else:
            a = "\n".join(a)
        x['answer'] = a
        return x
    ds = ds.map(format_answer)
    ds = ds.map(lambda x: {"problem": x.pop("context").strip() + "\n\n" + x.pop("query").strip(), "solution": x.pop("answer"), "domain": "linguistics", "source": "facebook/linguini", "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
    return ds

def load_deepcoder():
    # agentica-org/DeepCoder-Preview-Dataset
    dsc = datasets.load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset", "codeforces", split="test"
    ).map(lambda x: {"problem": x.pop("problem"), "solution": None, "tests": x.pop("tests"), "domain": "code", "source": "agentica-org/DeepCoder-Preview-Dataset/codeforces", "metadata": str(x)})
    dsc = dsc.remove_columns([c for c in dsc.column_names if c not in DS_COLUMNS])
    dsl = datasets.concatenate_datasets([
        datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5", split="train"),
        datasets.load_dataset("agentica-org/DeepCoder-Preview-Dataset", "lcbv5", split="test"),
    ]).map(lambda x: {"problem": x.pop("problem"), "solution": None, "tests": x.pop("tests"), "domain": "code", "source": "agentica-org/DeepCoder-Preview-Dataset/lcbv5", "metadata": str(x)})
    dsl = dsl.remove_columns([c for c in dsl.column_names if c not in DS_COLUMNS])
    dsp = datasets.load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset", "primeintellect", split="train"
    ).map(lambda x: {"problem": x.pop("problem"), "solution": x["solutions"][0], "tests": x.pop("tests"), "domain": "code", "source": "agentica-org/DeepCoder-Preview-Dataset/primeintellect", "metadata": str(x)})
    dsp = dsp.remove_columns([c for c in dsp.column_names if c not in DS_COLUMNS])
    dst = datasets.load_dataset(
        "agentica-org/DeepCoder-Preview-Dataset", "taco", split="train"
    ).map(lambda x: {"problem": x.pop("problem"), "solution": None, "tests": x.pop("tests"), "domain": "code", "source": "agentica-org/DeepCoder-Preview-Dataset/taco", "metadata": str(x)})
    dst = dst.remove_columns([c for c in dst.column_names if c not in DS_COLUMNS])
    ds = datasets.concatenate_datasets([dsc, dsl, dsp, dst])
    return ds

# def load_medicalo1():
#     # https://huggingface.co/datasets/FreedomIntelligence/medical-o1-verifiable-problem
#     ds = datasets.load_dataset("FreedomIntelligence/medical-o1-verifiable-problem", split="train")
#     ds.map(lambda x: {"problem": x.pop("Open-ended Verifiable Question"), "solution": x.pop("Ground-True Answer"), "domain": "medical", "source": "FreedomIntelligence/medical-o1-verifiable-problem", "metadata": str(x)})
#     ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS])
#     return ds
# Included in below

def merge_generalreasoning(ds):
    import pandas as pd
    df = ds.to_pandas()
    grouped_df = df.groupby('question').agg({
        'model_name': lambda x: list(x),
        'verifier_score': lambda x: list(x),
        'reference_answer': lambda x: x.iloc[0],
        'task': lambda x: x.iloc[0],
        'question_source': lambda x: x.iloc[0],
        'question_url': lambda x: x.iloc[0],
        'question_license': lambda x: x.iloc[0],
    }).reset_index()
    def add_difficulty(x):
        d = {}
        for m, v in zip(x['model_name'], x['verifier_score']):
            if m not in d:
                d[m] = str(int(v)) + "/1" ### TODO: This should be reversed i.e. getting it correct should be 0/1 in terms of difficulty
            else:
                k,n = d[m].split("/")
                d[m] = str(int(k) + int(v)) + "/" + str(int(n) + 1)
        return json.dumps(d) # Make it a string such that it can have different keys for different rows
    grouped_df['difficulty'] = grouped_df.apply(add_difficulty, axis=1)
    ds = datasets.Dataset.from_pandas(grouped_df)
    return ds

def load_generalreasoning():
    ds = datasets.load_dataset("GeneralReasoning/GeneralThought-430K")['train'] # 430788
    ds = ds.filter(lambda x: bool(x["reference_answer"])) # 305044
    ds = ds.filter(lambda x: x["verifier_score"] is not None) # 292774

    # Option 1: Don't add difficulty
    # ds = ds.filter(lambda x: x["verifier_score"] == 1) # 189751
    # ds = ds.filter(lambda x: len(x["reference_answer"]) < 600) # 176241
    # # Remove duplicate questions
    # df = ds.to_pandas()
    # df = df.drop_duplicates(subset="question")
    # ds = datasets.Dataset.from_pandas(df) # 160659
    # Option 2: Add difficulty
    ds = merge_generalreasoning(ds)
    ds = ds.filter(lambda x: sum([int(v.split("/")[0]) for v in json.loads(x["difficulty"]).values() if v is not None]) >= 1) # 173342
    ds = ds.filter(lambda x: len(x["reference_answer"]) < 600) # 160649
    # It took 21 samples until I found a good enough "databricks/databricks-dolly-15k" and even that one wasn't great
    # so 8401/21 = ~400 good samples which is probably not worth the difficulty of correctly catching all bad ones
    # most of its questions are subjective like "Who is the most sensational player in MLB today?" etc
    ds = ds.filter(lambda x: x["question_source"] != "databricks/databricks-dolly-15k") # 152248

    SOURCE_TO_DOMAIN = {
        'FreedomIntelligence/medical-o1-verifiable-problem': 'medical',
        'Numina/NuminaMath': 'math',
        'General/compmath': 'math',
        'Hendryks/MATH': 'math',
        'OpenAI/GSM8K': 'math',
    }

    ds = ds.map(lambda x: {"problem": x.pop("question"), "solution": x.pop("reference_answer"), "domain": SOURCE_TO_DOMAIN.get(x['question_source']), "source": "GeneralReasoning/GeneralThought-430K/" + x.pop('question_source'), "difficulty": x.pop("difficulty"), "metadata": str(x)})
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS + ["difficulty"]])
    return ds

def load_skywork():
    ds = datasets.load_dataset("Skywork/Skywork-OR1-RL-Data", split="math")
    ds = ds.map(lambda x: {
        "problem": x.pop("prompt")[0]["content"], # All are of len 1
        "solution": json.loads(x.pop("reward_model")['ground_truth'])[0], # All are of len 1
        "tests": None,
        "domain": x.pop("ability"),
        "source": "Skywork/Skywork-OR1-RL-Data" + "/" + x.pop("data_source"),
        "difficulty": json.dumps({k: str(v) + "/16" for k,v in x.pop("extra_info").pop("model_difficulty").items()}),
        "metadata": str(x)}
    )
    dsc = datasets.load_dataset("Skywork/Skywork-OR1-RL-Data", split="code")
    dsc = dsc.map(lambda x: {
        "problem": x.pop("prompt")[0]["content"], # All are of len 1
        # https://huggingface.co/datasets/Skywork/Skywork-OR1-RL-Data/viewer/default/code?row=14
        "solution": None,
        "tests": x.pop("reward_model")['ground_truth'],
        "domain": x.pop("ability"),
        "source": "Skywork/Skywork-OR1-RL-Data" + "/" + x.pop("data_source"),
        "difficulty": json.dumps({k: str(v) + "/16" for k,v in x.pop("extra_info").pop("model_difficulty").items()}),
        "metadata": str(x)}
    )
    ds = datasets.concatenate_datasets([ds, dsc])
    ds = ds.remove_columns([c for c in ds.column_names if c not in DS_COLUMNS + ["difficulty"]])
    return ds

def decontaminate_train_data(train_questions, test_questions, ds, ngram_size=8):    
    # Build ngram lookups
    train_lookup = build_ngram_lookup(train_questions, ngram_size)
    test_lookup = build_ngram_lookup(test_questions, ngram_size)

    # Find contaminated questions
    contaminated_ids = find_contaminated_questions(train_lookup, test_lookup)

    # Remove contaminated examples
    not_contaminated_ids = set(range(len(train_questions))) - contaminated_ids
    ds = ds.select(list(not_contaminated_ids))
    print(f"\nDecontamination Results:")
    print(f"Total train questions: {len(train_questions)}")
    print(f"Contaminated questions: {len(contaminated_ids)}")
    print(f"Contamination rate: {(len(contaminated_ids)/len(train_questions)*100):.2f}%")
    print(f"Clean examples remaining: {len(ds)}")
    return ds

def dedup_train_data(train_questions, ds, ngram_size=8):
    # Build ngram lookups
    train_lookup = build_ngram_lookup(train_questions, ngram_size)
    # Find duplicate document IDs
    duplicate_ids = set()
    for doc_ids in train_lookup.values():
        if len(doc_ids) > 1:  # If the same n-gram exists in multiple documents
            # Add all but the lowest occurrence to duplicates
            duplicate_ids.update(sorted(doc_ids)[1:])

    # Keep only non-duplicate IDs
    unique_ids = set(range(len(train_questions))) - duplicate_ids
    ds = ds.select(list(unique_ids))
    
    print(f"\nDeduplication Results:")
    print(f"Total train questions: {len(train_questions)}")
    print(f"Duplicate questions: {len(duplicate_ids)}")
    print(f"Duplication rate: {(len(duplicate_ids)/len(train_questions)*100):.2f}%")
    print(f"Unique examples remaining: {len(ds)}")
    return ds

DS_TO_SELECTION = {
    # Name: [load function, selection function, #samples]

    # Skywork - high quality with difficulty so put at top
    "Skywork": [load_skywork, None, None],
    # General reasoning - high quality with difficulty so put at top
    "GeneralReasoning": [load_generalreasoning, None, None],
    # High quality code
    "DeepCoder": [load_deepcoder, None, None],
    # Very high-quality so take all (12K)
    "MATH": [load_math, None, None],
    # Very high-quality so take all (3922)
    "OlympicArena": [load_olympic_arena, None, None],
    # Take all (720)
    "TheoremQA": [load_theoremqa, None, None],
    # Take all as super high-quality (3329)
    "Omni-MATH": [load_omnimath, None, None],
    # Very high-quality so take all (626)
    "OlympiadBench": [load_olympiad_bench, None, None],
    # Very high-quality so take all (483)
    "JEEBench": [load_jeebench, None, None],
    # Very high-quality so take all
    # Includes data from MATH test set so most gets removed in decontam
    "AGIEval": [load_agieval, None, None],
    # Very high-quality so take all
    "GPQA": [load_gpqa_extended, None, None],
    # High-quality
    "XWord": [load_xword, None, None],
    # Very high-quality so take all (520)
    # "USACO": [load_usaco, None, None], # Probably enough code for now
    # Very high-quality so take all (24)
    "s1teasers": [load_s1teasers, None, None],
    # Very high-quality so take all
    # "LiveCodeBench": [load_livecodebench, None, None], # Include in code below
    # Pretty big (~900K) but take all due to stringent filters that leave ~250K
    # ~900 * 0.7 * 0.943 * 0.959 * 0.5 = ~284 (filtering out proofs incomplete etc)
    "NuminaMath": [load_numinamath, None, None],

    ### New datasets for s2 ###
    # Maybe HLE
    # Maybe BBH # Maybe not hard enough as Qwen2.5-1.5B gets 45% https://arxiv.org/pdf/2412.15115 (vs 35% on MATH)
    # https://huggingface.co/datasets/SynthLabsAI/Big-Math-RL-Verified - likely a subset of NuminaMath
    # Maybe AMC problems though a bit easy
    # Maybe https://arxiv.org/abs/2504.16074 though too hard for current 1.5B models, maybe for 32B
    # Maybe https://huggingface.co/datasets/nvidia/OpenCodeReasoning
    # Maybe https://huggingface.co/datasets/nvidia/OpenMathReasoning
    # Maybe SimpleQA
}

if __name__ == "__main__":
    random.seed(42)
    # Load test problems
    test_problems = (
        datasets.load_dataset("simplescaling/aime24_nofigures", split="train")["problem"] +
        datasets.load_dataset("simplescaling/aime25_nofigures", split="train")["problem"] +
        datasets.load_dataset("simplescaling/openaimath", split="test")["problem"] + 
        datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True)["Question"] #  +
        # load_dataset(
        #         "livecodebench/code_generation_lite", split="test", version_tag="v4_v5", trust_remote_code=True
        #     )["question_content"]
    )

    ds_all = []
    for ds_name, (load_fn, selection_fn, n_samples) in DS_TO_SELECTION.items():
        print(f"Processing {ds_name}...")
        ds = load_fn()
        ds = decontaminate_train_data(ds['problem'], test_problems, ds, ngram_size=12)
        ds = ds.shuffle(seed=42)
        if n_samples:
            ds = ds.select(range(n_samples))
        test_problems += ds['problem']
        if "tests" not in ds.column_names:
            new_column = [None] * len(ds)
            ds = ds.add_column("tests", new_column)
        if "difficulty" not in ds.column_names:
            new_column = [None] * len(ds)
            ds = ds.add_column("difficulty", new_column)
        ds_all.append(ds)
    ds = datasets.concatenate_datasets(ds_all)

    # Exact deduplication
    memory = set()
    def is_unique(elem, column, memory):
        if elem[column] in memory: return False
        memory.add(elem[column])
        return True
    ds = ds.filter(partial(is_unique, column="problem", memory=memory))
    # Ngram deduplication
    ds_nodeup = ds.filter(lambda x: x["source"].startswith("Skywork") and x["domain"] == "code")
    ds_dedup = ds.filter(lambda x: not(x["source"].startswith("Skywork") and x["domain"] == "code"))
    ds_dedup = dedup_train_data(ds_dedup['problem'], ds_dedup, ngram_size=16)
    ds = datasets.concatenate_datasets([ds_nodeup, ds_dedup])

    ds.push_to_hub("prefixsliding/train_v1")
