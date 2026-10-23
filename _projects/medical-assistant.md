---
layout: page
title: Medical Assistant
description: "An assistant that reads the medical literature for a doctor: entities found in articles, mapped to medical vocabularies, stored as a graph and ranked by the strength of the evidence behind them"
img: assets/img/projects/1_medical_assistant/figures/card.png
cover: /assets/img/projects/1_medical_assistant/figures/cover.png
og_image: https://xmarva.github.io/assets/img/projects/1_medical_assistant/figures/cover.png
stylesheets: [/assets/css/tfs-course.css]
importance: 5
category: work
related_publications: false

role: Data Scientist

domains:
  - NLP
  - Biomed
  - NeuroSymbolic AI

tech:
  - NER
  - Medical Ontologies
  - Graph Databases
  - Language Models

website: http://www.meanotek.io
---

{% raw %}

In 2002 a group of doctors counted how much a primary care physician would have to read to keep up. They took the 341 journals relevant to primary care and found 7,287 new articles a month. Reading them with the care an epidemiologist would take, they estimated, would cost "627.5 hours per month, or about 29 hours per weekday" ([Alper et al., 2004](https://pmc.ncbi.nlm.nih.gov/articles/PMC521514/)).

That was one specialty, sixteen years before I worked on this problem. In the 2018 fiscal year PubMed added 1,329,148 citations, about 3,600 a day ([NLM](https://www.nlm.nih.gov/bsd/medline_pubmed_production_stats.html)).

In 2017 and 2018 I worked as a data scientist at Meanotek, a small company building language technology, on an assistant that would do this reading for a doctor. Ask it about a disease, and it shows the treatments, the symptoms, the related conditions and the studies behind each claim, with an honest note on how strong those studies are.

The project's screenshots have not survived, so the screen below is drawn again for this page. The numbers on it are real: PubMed counts up to 2018 and the links of a public 2017 knowledge graph.

<img class="img-fluid rounded d-block mx-auto my-4" src="/assets/img/projects/1_medical_assistant/figures/interface.png" width="760" height="480" alt="A clinician's screen for type 2 diabetes mellitus. A header with its codes ICD-10 E11, SNOMED CT 44054006 and MeSH D003924. A table of treatments ranked by evidence, with the number of randomized trials and meta-analyses in PubMed up to 2018: metformin 1,531 trials and 212 meta-analyses, pioglitazone 459 and 70, sitagliptin 295 and 77, glimepiride 242 and 17, acarbose 162 and 13, linagliptin 87 and 26, gliclazide 107 and 14. The Oxford levels of evidence. On the right, symptoms the disease presents with, diseases it resembles, and the newest trials on metformin from December 2018" loading="lazy" style="border: 1px solid var(--global-divider-color)">

<p class="tfs-caption">The assistant's main screen, reconstructed. Counts: PubMed, publication years up to 2018. Symptoms and similar diseases: Hetionet v1.0.</p>

## How fast the literature grows

<img class="tfs-figure tfs-light" src="/assets/img/projects/1_medical_assistant/figures/growth-light.png" width="760" alt="Two panels. Left: bars of PubMed articles per year of publication from 1990 to 2018, growing from 0.41 million to 1.35 million. Right: lines of randomized trials, from 6,760 to 26,686 a year, and meta-analyses, from 271 to 14,407" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/1_medical_assistant/figures/growth-dark.png" width="760" alt="Two panels. Left: bars of PubMed articles per year of publication from 1990 to 2018, growing from 0.41 million to 1.35 million. Right: lines of randomized trials, from 6,760 to 26,686 a year, and meta-analyses, from 271 to 14,407" loading="lazy">

The growth is not only in volume. The kinds of studies a doctor trusts most grew faster than everything else: meta-analyses, which pool the results of many trials, went from a few hundred a year in 1990 to over fourteen thousand in 2018.

That is good news and a problem at once. The best evidence on a question now exists more often, but it sits among a million other articles a year, and a search box returns them all in the same list.

## Finding the entities

The first step is to find what an article talks about: diseases, symptoms, drugs, doses, outcomes. This is named entity recognition, and for medical text it starts with a decision about what counts.

Is "lower risk of hypoglycemia" a mention of a disease? Is "insulin glargine" one drug or a drug and a modifier? Is "type 2" part of the name? I wrote the annotation guidelines that answered questions like these, because a model can only be as consistent as the people who label its training data.

<img class="tfs-figure tfs-light" src="/assets/img/projects/1_medical_assistant/figures/ner-light.png" width="760" alt="A sentence from a 2018 trial abstract split into tokens, each with a BIO tag below it. Linagliptin is B-DRUG; insulin glargine is B-DRUG I-DRUG; hypoglycemia is B-DIS, for disease; type 2 diabetes is B-DIS I-DIS I-DIS; every other token is O. Under each entity is the identifier it is normalised to: RxNorm 1100699 for linagliptin, RxNorm 274783 for insulin glargine, MeSH D007003 for hypoglycemia, and MeSH D003924 and ICD-10 E11 for type 2 diabetes" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/1_medical_assistant/figures/ner-dark.png" width="760" alt="A sentence from a 2018 trial abstract split into tokens, each with a BIO tag below it. Linagliptin is B-DRUG; insulin glargine is B-DRUG I-DRUG; hypoglycemia is B-DIS, for disease; type 2 diabetes is B-DIS I-DIS I-DIS; every other token is O. Under each entity is the identifier it is normalised to: RxNorm 1100699 for linagliptin, RxNorm 274783 for insulin glargine, MeSH D007003 for hypoglycemia, and MeSH D003924 and ICD-10 E11 for type 2 diabetes" loading="lazy">

The usual format is BIO: every token is tagged as the beginning of an entity, inside one, or outside. The tagger of choice in 2017 was a bidirectional LSTM with a conditional random field on top, the CRF making sure an "inside" tag never follows an "outside" one ([Huang et al., 2015](https://arxiv.org/abs/1508.01991); [Lample et al., 2016](https://aclanthology.org/N16-1030)).

Habibi and colleagues showed in 2017 that one generic LSTM-CRF, fed with word vectors trained on PubMed, beat entity-specific tools on 33 biomedical datasets: "On average, F1-score of LSTM-CRF is 5% above that of the baselines, mostly due to a sharp increase in recall" ([Habibi et al., 2017](https://pubmed.ncbi.nlm.nih.gov/28881963/)). The vectors themselves were trained on 5.5 billion tokens of PubMed and PubMed Central ([Pyysalo et al., 2013](http://bio.nlplab.org/)).

The public training data was small. The NCBI disease corpus has 793 PubMed abstracts with 6,892 disease mentions ([Doğan et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24393765/)); BC5CDR adds 1,500 articles with chemicals and diseases ([Li et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27161011/)). For a product, the rest of the labels had to come from our own annotation.

The field moved fast around us. ELMo, with word representations that depend on the sentence, appeared in February 2018 ([Peters et al., 2018](https://arxiv.org/abs/1802.05365)), and BERT in October 2018 ([Devlin et al., 2018](https://arxiv.org/abs/1810.04805)). BioBERT, trained on PubMed, came out in January 2019, after this project ([Lee et al., 2019](https://arxiv.org/abs/1901.08746)).

## Many names for one thing

Finding "heart attack" in a sentence is half the job. The other half is knowing that it is the same thing as "myocardial infarction", "MI" and "acute MI", and that "Glucophage" is metformin.

<img class="tfs-figure tfs-light" src="/assets/img/projects/1_medical_assistant/figures/normalize-light.png" width="760" alt="Two examples of normalisation. The phrases heart attack, MI, myocardial infarction and acute MI all map to one concept, myocardial infarction, which has SNOMED CT 22298006, MeSH D009203, UMLS C0027051 and, for the acute case only, ICD-10 I21. The phrases Glucophage, metformin HCl, metformin 500 mg and metformin map to the ingredient metformin: RxNorm 6809, DrugBank DB00331, ATC A10BA02, MeSH D008687" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/1_medical_assistant/figures/normalize-dark.png" width="760" alt="Two examples of normalisation. The phrases heart attack, MI, myocardial infarction and acute MI all map to one concept, myocardial infarction, which has SNOMED CT 22298006, MeSH D009203, UMLS C0027051 and, for the acute case only, ICD-10 I21. The phrases Glucophage, metformin HCl, metformin 500 mg and metformin map to the ingredient metformin: RxNorm 6809, DrugBank DB00331, ATC A10BA02, MeSH D008687" loading="lazy">

Medicine has vocabularies for this, more than one. The Unified Medical Language System links them: its 2018 release held "approximately 3.67 million concepts and 14 million unique concept names from 203 source vocabularies" ([NLM, 2018](https://www.nlm.nih.gov/pubs/techbull/mj18/mj18_umls_2018aa_release.html)). SNOMED CT alone had 340,659 active clinical concepts in July 2018 ([SNOMED International](http://web.archive.org/web/20181007171143/http://www.snomed.org/snomed-ct/five-step-briefing)).

Drugs have their own system, RxNorm, "a normalized naming system for generic and branded drugs" ([NLM](https://www.nlm.nih.gov/research/umls/rxnorm/overview.html)). It is why a dose, a salt and a brand name can all be traced to one ingredient.

The vocabularies do not map one to one, and the differences matter. SNOMED CT has one concept for myocardial infarction; ICD-10, the classification hospitals used then, splits it into acute (I21), subsequent (I22) and old (I25.2). WHO released ICD-11 in June 2018, but as "an advance preview", to come into effect in 2022 ([WHO, 2018](https://www.who.int/news/item/18-06-2018-who-releases-new-international-classification-of-diseases-%28icd-11%29)).

Dictionary tools did the matching before neural taggers did: MetaMap from the National Library of Medicine ([Aronson, 2001](https://pubmed.ncbi.nlm.nih.gov/11825149/)), and QuickUMLS, "up to 135 times faster" ([Soldaini and Goharian, 2016](http://ir.cs.georgetown.edu/downloads/quickumls.pdf)). They are precise on exact names and lost on everything else, which is where a trained tagger helps and the two are best combined.

## A graph instead of tables

<img class="img-fluid rounded d-block mx-auto my-4" src="/assets/img/projects/1_medical_assistant/2.png" width="2520" height="1300" alt="The Neo4j browser showing a star-shaped graph of 227 nodes with one disease node, gastritis, in the centre, and the others around it coloured by type: drugs, drug classes, substances, symptoms, symptom groups, definitions and identifiers" loading="lazy">

<p class="tfs-caption">One disease in the project's graph, in the Neo4j browser: gastritis at the centre of 227 nodes, coloured by type.</p>

Once entities are found and named, the question is where to keep them. Medical knowledge is mostly relations: a drug treats a disease, belongs to a class, interacts with another drug; a disease presents a symptom, resembles another disease. In a relational database each of these is a join table; in a graph database it is just an edge.

We kept the entities in Neo4j, "an open-source NoSQL native graph database" that had been public since 2007 ([Neo4j, 2018](http://web.archive.org/web/20180307201952/https://neo4j.com/developer/graph-database/)). A question like "which drugs of this class treat diseases that resemble this one" becomes a short pattern in its query language, Cypher.

The idea was not ours alone. In 2017 Himmelstein and colleagues published Hetionet, a graph of "47,031 nodes of 11 types and 2,250,197 relationships of 24 types" integrated from 29 public resources, and served it as a Neo4j database anyone could query ([Himmelstein et al., 2017](https://elifesciences.org/articles/26726)).

<img class="tfs-figure tfs-light" src="/assets/img/projects/1_medical_assistant/figures/graph-light.png" width="760" alt="A radial graph around type 2 diabetes from Hetionet: 22 compounds that treat it, including metformin, sitagliptin, pioglitazone and glimepiride; 14 symptoms that co-occur with it in MEDLINE, such as obesity, albuminuria and gastroparesis; and 8 diseases it resembles, such as type 1 diabetes, metabolic syndrome and polycystic ovary syndrome" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/1_medical_assistant/figures/graph-dark.png" width="760" alt="A radial graph around type 2 diabetes from Hetionet: 22 compounds that treat it, including metformin, sitagliptin, pioglitazone and glimepiride; 14 symptoms that co-occur with it in MEDLINE, such as obesity, albuminuria and gastroparesis; and 8 diseases it resembles, such as type 1 diabetes, metabolic syndrome and polycystic ovary syndrome" loading="lazy">

The figure is type 2 diabetes in Hetionet, and it shows a trap of building such a graph from text. The symptoms come from co-occurrence in MEDLINE, so next to obesity and albuminuria sit birth weight and flushing: words that often appear in the same articles, not signs a doctor would look for. A graph built by counting needs a filter, and a doctor needs to see where each edge came from.

## How much to trust a study

An assistant that lists every study with equal weight is a search engine. The useful part is telling a meta-analysis of randomized trials from a case report.

Evidence-based medicine has a ladder for this. In the Oxford levels of 2011, a systematic review of randomized trials is level 1 for questions of treatment, a single randomized trial is level 2, and "mechanism-based reasoning" is level 5 ([CEBM, 2011](http://web.archive.org/web/20181227145516/https://www.cebm.net/2016/05/ocebm-levels-of-evidence/)). GRADE rates the quality of a body of evidence as high, moderate, low or very low, and starts observational studies at low ([Guyatt et al., 2008](https://pmc.ncbi.nlm.nih.gov/articles/PMC2335261/)).

<img class="tfs-figure tfs-light" src="/assets/img/projects/1_medical_assistant/figures/evidence-light.png" width="760" alt="Horizontal bars of PubMed articles on metformin in type 2 diabetes up to 2018, by publication type, with the Oxford levels of evidence: meta-analyses 212, randomized trials 1,531, clinical trials 1,764, observational studies 166, comparative studies 862, case reports 281, reviews 1,728" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/1_medical_assistant/figures/evidence-dark.png" width="760" alt="Horizontal bars of PubMed articles on metformin in type 2 diabetes up to 2018, by publication type, with the Oxford levels of evidence: meta-analyses 212, randomized trials 1,531, clinical trials 1,764, observational studies 166, comparative studies 862, case reports 281, reviews 1,728" loading="lazy">

PubMed labels articles with publication types, and those labels are the first signal. They come late: Marshall and colleagues measured "a lag time of around 250 days from publication until the PT tag is applied", and trained a classifier to recognize randomized trials from their text instead ([Marshall et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6030513/)).

The newest studies, the ones a doctor most wants to hear about, are exactly the ones without labels. We classified publications with classical machine learning on what could be measured about them: the design of the study, the quality of its data, how often it was cited, the record of its authors. RobotReviewer had shown that even the risk of bias in a trial could be judged by a model with useful accuracy ([Marshall et al., 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC4713900/)).

The cautionary example of the time was IBM's Watson for Oncology. In September 2017 STAT reported that its recommendations were "derived from the training provided by the hospital's doctors, not the outside literature" ([Ross and Swetlitz, 2017](https://www.statnews.com/2017/09/05/watson-ibm-cancer/)). In July 2018 it reported internal documents citing "multiple examples of unsafe and incorrect treatment recommendations" ([Ross and Swetlitz, 2018](https://www.statnews.com/2018/07/25/ibm-watson-recommended-unsafe-incorrect-treatments/)).

It is the reason the assistant was built to show evidence and its strength rather than to recommend, and to leave the decision to the doctor.

## The system

<img class="tfs-figure tfs-light" src="/assets/img/projects/1_medical_assistant/figures/pipeline-light.png" width="760" alt="The pipeline in two rows. Sources (PubMed, trial registries, drug databases) are split into sentences and tokens; a BiLSTM-CRF tagger finds entities; they are mapped to concepts in UMLS, SNOMED CT and RxNorm; relations between them are extracted. The result goes into a Neo4j graph queried with Cypher; each study is graded by its type and quality, answers are ranked per question and served through a Django REST API to the doctor&#x27;s screen" loading="lazy">
<img class="tfs-figure tfs-dark" src="/assets/img/projects/1_medical_assistant/figures/pipeline-dark.png" width="760" alt="The pipeline in two rows. Sources (PubMed, trial registries, drug databases) are split into sentences and tokens; a BiLSTM-CRF tagger finds entities; they are mapped to concepts in UMLS, SNOMED CT and RxNorm; relations between them are extracted. The result goes into a Neo4j graph queried with Cypher; each study is graded by its type and quality, answers are ranked per question and served through a Django REST API to the doctor&#x27;s screen" loading="lazy">

Behind the screen, sources were downloaded and parsed, entities and relations were extracted, and the results went into two stores: PostgreSQL for the articles and their metadata, and Neo4j for the entities and their links. A Django service answered the screen's questions over a REST API.

## References

Literature and evidence:

- Alper et al. [How much effort is needed to keep up with the literature relevant for primary care?](https://pmc.ncbi.nlm.nih.gov/articles/PMC521514/) J Med Libr Assoc, 2004.
- Guyatt et al. [GRADE: an emerging consensus on rating quality of evidence and strength of recommendations](https://pmc.ncbi.nlm.nih.gov/articles/PMC2335261/). BMJ, 2008.
- OCEBM Levels of Evidence Working Group. [The Oxford 2011 Levels of Evidence](http://web.archive.org/web/20181227145516/https://www.cebm.net/2016/05/ocebm-levels-of-evidence/).
- Marshall et al. [RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials](https://pmc.ncbi.nlm.nih.gov/articles/PMC4713900/). JAMIA, 2016.
- Marshall et al. [Machine learning for identifying randomized controlled trials](https://pmc.ncbi.nlm.nih.gov/articles/PMC6030513/). Research Synthesis Methods, 2018.

Entities and vocabularies:

- Huang, Xu and Yu. [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991). 2015.
- Lample et al. [Neural Architectures for Named Entity Recognition](https://aclanthology.org/N16-1030). NAACL 2016.
- Habibi et al. [Deep learning with word embeddings improves biomedical named entity recognition](https://pubmed.ncbi.nlm.nih.gov/28881963/). Bioinformatics, 2017.
- Doğan, Leaman and Lu. [NCBI disease corpus](https://pubmed.ncbi.nlm.nih.gov/24393765/). J Biomed Inform, 2014.
- Li et al. [BioCreative V CDR task corpus](https://pubmed.ncbi.nlm.nih.gov/27161011/). Database, 2016.
- Aronson. [Effective mapping of biomedical text to the UMLS Metathesaurus: the MetaMap program](https://pubmed.ncbi.nlm.nih.gov/11825149/). AMIA, 2001.
- Soldaini and Goharian. [QuickUMLS](http://ir.cs.georgetown.edu/downloads/quickumls.pdf). MedIR workshop, SIGIR 2016.

Graphs:

- Himmelstein et al. [Systematic integration of biomedical knowledge prioritizes drugs for repurposing](https://elifesciences.org/articles/26726). eLife, 2017. [Data](https://github.com/hetio/hetionet).

{% endraw %}
