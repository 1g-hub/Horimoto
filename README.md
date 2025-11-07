# Horimoto

ほりぃのファイル

Knowledge Graph の研究

ナレッジグラフ推論チャレンジへの挑戦

Marvel 映画は現在落ち気味 (20231005)

Marvel 映画の来年公開 Avengers: Doomsday が楽しみ（20251106）

## ディレクトリの構成

<pre>
.
├── B3
│   ├── jikken2_20221115_horimoto.pdf
│   └── jikken2_20221122_horimoto.pdf
├── B4
│   ├── 2023_1st_horimoto.pdf
│   ├── 2023_2nd_horimoto.pdf
│   ├── Summary
│   │   ├── KG-BERT.png
│   │   ├── KG-MLM.png
│   │   ├── KG-MLM1.png
│   │   ├── KG-MLM2.png
│   │   ├── abstract.bib
│   │   ├── abstract.sty
│   │   ├── jabbrvunsrt.bst
│   │   ├── latexmkrc
│   │   └── main.tex
│   ├── Summary_4_1201201120_堀本隆誠.pdf
│   ├── Thesis
│   │   ├── Thesis_4_1201201120_堀本隆誠.pdf
│   │   ├── assets
│   │   │   ├── BERT_fine-tuning.png
│   │   │   ├── Ex_KG.png
│   │   │   ├── KG-BERT.png
│   │   │   ├── KG-MLM.png
│   │   │   ├── KG-MLM1.png
│   │   │   ├── KG-MLM2.png
│   │   │   ├── MLM_fine-tuning_English.png
│   │   │   └── Transformer.png
│   │   ├── ccaption.sty
│   │   ├── doc
│   │   │   ├── bibliography.tex
│   │   │   ├── conclusion.tex
│   │   │   ├── experiment.tex
│   │   │   ├── introduction.tex
│   │   │   ├── propose.tex
│   │   │   ├── shaji.tex
│   │   │   └── tech.tex
│   │   ├── iepaper.sty
│   │   ├── ilib.sty
│   │   ├── index.bib
│   │   ├── index.tex
│   │   ├── iparam.sty
│   │   ├── latexmkrc
│   │   ├── oddchar.sty
│   │   └── soturon.sty
│   └── Thesis_4_1201201120_堀本隆誠.pdf
├── M1
│   ├── 2024_1st_horimoto.pdf
│   ├── 2024_1st_horimoto.pptx
│   ├── 2024_1st_slide_horimoto.pdf
│   └── README.md
├── README.md
├── Weekly_report
│   ├── B4
│   │   ├── NLP_2023_04_24_horimoto.pdf
│   │   ├── NLP_2023_05_01_horimoto.pdf
│   │   ├── NLP_2023_05_08_horimoto.pdf
│   │   ├── NLP_2023_05_15_horimoto.pdf
│   │   ├── NLP_2023_05_22_horimoto.pdf
│   │   ├── NLP_2023_05_29_horimoto.pdf
│   │   ├── NLP_2023_06_05_horimoto.pdf
│   │   ├── NLP_2023_06_12_horimoto.pdf
│   │   ├── NLP_2023_06_19_horimoto.pdf
│   │   ├── NLP_2023_07_10_horimoto.pdf
│   │   ├── NLP_2023_07_24_horimoto.pdf
│   │   ├── NLP_2023_07_31_horimoto.pdf
│   │   ├── NLP_2023_08_07_horimoto.pdf
│   │   ├── NLP_2023_08_28_horimoto.pdf
│   │   ├── NLP_2023_09_04_horimoto.pdf
│   │   ├── NLP_2023_09_11_horimoto.pdf
│   │   ├── NLP_2023_09_25_horimoto.pdf
│   │   ├── NLP_2023_10_02_horimoto.pdf
│   │   ├── NLP_2023_10_09_horimoto.pdf
│   │   ├── NLP_2023_10_16_horimoto.pdf
│   │   ├── NLP_2023_10_30_horimoto.pdf
│   │   ├── NLP_2023_11_06_horimoto.pdf
│   │   ├── NLP_2023_11_13_horimoto.pdf
│   │   ├── NLP_2023_11_27_horimoto.pdf
│   │   ├── NLP_2023_12_04_horimoto.pdf
│   │   ├── NLP_2023_12_11_horimoto.pdf
│   │   ├── NLP_2024_01_10_horimoto.pdf
│   │   ├── NLP_2024_01_15_horimoto.pdf
│   │   ├── NLP_2024_01_22_horimoto.pdf
│   │   ├── NLP_2024_01_29_horimoto.pdf
│   │   ├── NLP_2024_03_25_horimoto.pdf
│   │   ├── NLP_2024_04_08_horimoto.pdf
│   │   ├── NLP_2024_04_15_horimoto.pdf
│   │   └── NLP_2024_05_20_horimoto.pdf
│   └── M1
│       ├── NLP_2024_04_08_horimoto.pdf
│       └── NLP_2024_06_10_horimoto.pdf
├── 学会
│   └── JSAI2024
│       ├── 2024_JSAI.pptx
│       ├── JSAI_2024.pdf
│       ├── assets
│       │   ├── Ex_KG.png
│       │   ├── KG-BERT.png
│       │   ├── KG-MLM.png
│       │   ├── KG-MLM1.png
│       │   ├── KG-MLM2.png
│       │   ├── MLM_fine-tuning_English.png
│       │   └── MLM_fine_tuning_Japanese.png
│       ├── index.bib
│       ├── jsai.bst
│       ├── jsaiac.sty
│       ├── latexmkrc
│       └── main.tex
└── 研究
    └── 学士
        └── KG-BERT
            ├── data.py
            ├── e.sh
            ├── mlm_result
            │   ├── output_tail_1
            │   │   ├── 1
            │   │   │   ├── predict_tail.txt
            │   │   │   ├── rank_triples.txt
            │   │   │   └── result.txt
            │   │   ├── 2
            │   │   │   ├── predict_tail.txt
            │   │   │   ├── rank_triples.txt
            │   │   │   └── result.txt
            │   │   └── 3
            │   │       ├── predict_tail.txt
            │   │       ├── rank_triples.txt
            │   │       └── result.txt
            │   └── output_tail_caption_sub_1
            │       ├── 1
            │       │   ├── predict_tail.txt
            │       │   └── rank_triples.txt
            │       ├── 2
            │       │   ├── predict_tail.txt
            │       │   ├── rank_triples.txt
            │       │   └── result.txt
            │       └── 3
            │           ├── predict_tail.txt
            │           ├── rank_triples.txt
            │           └── result.txt
            ├── mlm_tail.py
            ├── mlm_tail_caption.py
            ├── mlm_tail_caption_sub.py
            ├── preprocessing.py
            ├── requirements.txt
            ├── run_bert_entity_prediction.py
            ├── run_bert_link_prediction.py
            ├── run_bert_mlm.py
            ├── run_bert_relation_prediction.py
            ├── run_bert_triple_classifier.py
            └── wiki.py
</pre>


