import os, io, re, json
import pymupdf

SP = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SP, "txt")
os.makedirs(OUT, exist_ok=True)
D = "docs"

MAP = {
    "yan2014": "CASME II An Improved Spontaneous Micro-Expression.pdf",
    "liX2013": "SMIC A_Spontaneous_Micro-expression_Database_Inducement_collection_and_baseline.pdf",
    "davison2018": "SAMM_A_Spontaneous_Micro-Facial_Movement_Dataset.pdf",
    "qu2016": "CASME2_ADatabaseofSpontaneousMacro-expressionsandMicro-expressions.pdf",
    "see2019": "megc-2019-the-second-facial-micro-expressions-grand-challenge.pdf",
    "liX2018": "towards-reading-hidden-emotions-a-comparative-study-of-spontaneous-micro-expression-spotting-and-recognition-methods.pdf",
    "ben2021": "video-based-facial-micro-expression-analysis-a-survey-of-datasets-features-and-algorithms.pdf",
    "zhaoG2007": "Dynamic_Texture_Recognition_Using_Local_Binary_Patterns_with_an_Application_to_Facial_Expressions.pdf",
    "shreve2011": "Macro-and micro-expression spotting in long videos using spatio-temporal strain.pdf",
    "liong2014a": "Optical strain based recognition of subtle emotions.pdf",
    "liong2014b": "Subtle expression recognition using optical strain weighted features.pdf",
    "liong2016": "Spontaneous subtle expression detection and recognition based on facial strain.pdf",
    "lu2015": "A Delaunay-Based Temporal Coding Model for Micro-expression Recognition-703-716.pdf",
    "xu2017": "microexpression-identification-and-categorization-using-a-facial-dynamics-map.pdf",
    "liong2018": "less-is-more-micro-expression-recognition-from-video-using-apex-frame.pdf",
    "liong2019a": "off-apexnet-on-micro-expression-recognition-system.pdf",
    "liY2018": "can-micro-expression-be-recognized-based-on-single-apex-frame.pdf",
    "liY2021": "joint-local-and-global-information-learning-with-single-apex-frame-detection-for-micro-expression-recognition.pdf",
    "bai2021": "Micro-expression recognition based on video motion magnification and pre-trained neural network.pdf",
    "liong2019b": "shallow-triple-stream-three-dimensional-cnn-ststnet-for-micro-expression-recognition.pdf",
    "liu2019": "a-neural-micro-expression-recognizer.pdf",
    "zhaoS2021": "a-two-stage-3d-cnn-based-learning-method-for-spontaneous-micro-expression-recognition.pdf",
    "xia2020a": "spatiotemporal-recurrent-convolutional-networks-for-recognizing-spontaneous-micro-expressions.pdf",
    "xia2020b": "revealing-the-invisible-with-model-and-data-shrinking-for-composite-database-micro-expression-recognition.pdf",
    "zhang2022": "Short and Long Range Relation Based Spatio-Temporal Transformer for Micro-Expression Recognition.pdf",
    "dosovitskiy2021": "an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale.pdf",
    "yang2021": "SimAM A Simple, Parameter Free Attention Module for Convolutional Neural Networks.pdf",
    "xie2022": "An_Overview_of_Facial_Micro-Expression_Analysis_Data_Methodology_and_Challenge.pdf",
    "adegun2020": "Facial micro-expression recognition A Machine learning approach.pdf",
    "li2022survey": "Deep learning for micro-expression recognition A survey.pdf",
}

meta = {}
for key, fn in MAP.items():
    p = os.path.join(D, fn)
    if not os.path.exists(p):
        print("MISSING PDF", key, fn)
        continue
    doc = pymupdf.open(p)
    full = "\n".join(pg.get_text() for pg in doc)
    first = doc[0].get_text()
    io.open(os.path.join(OUT, key + ".txt"), "w", encoding="utf-8").write(full)
    meta[key] = {"pages": doc.page_count, "chars": len(full),
                 "first": re.sub(r"\s+", " ", first)[:600],
                 "pdfmeta": {k: v for k, v in doc.metadata.items() if v}}
    print("%-16s pages=%-3d chars=%-7d" % (key, doc.page_count, len(full)))

json.dump(meta, io.open(os.path.join(SP, "pdfmeta.json"), "w", encoding="utf-8"),
          indent=1, ensure_ascii=False)
print("done", len(meta))
