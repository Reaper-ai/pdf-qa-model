"""Build eval/gold_set.jsonl — 40 hand-written questions over the 5-paper corpus.

Each record pins the correct answer plus the exact source page and a verbatim
snippet of that page. Snippets are extracted from the production-cleaned page
text (TextNormalizer) so they match what the pipeline actually indexes.
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))

A = "attention_is_all_you_need.pdf"
B = "bert.pdf"
R = "retrieval_augmented_generation.pdf"
L = "lora.pdf"
D = "deep_residual_learning.pdf"

# (id, question, answer, doc, page, snippet_regex, type)
QUESTIONS = [
    # ---------------- Attention Is All You Need (8) ----------------
    ("q01", "What BLEU score did the Transformer big model achieve on the WMT 2014 English-to-German translation task?",
     "28.4 BLEU", A, 1, r"Our model achieves 28\.4 BLEU on the WMT 2014", "numeric"),
    ("q02", "How many days did it take to train the model that achieved a 41.8 BLEU score on English-to-French, and on how many GPUs?",
     "3.5 days on eight GPUs", A, 1, r"score of 41\.8 after training for 3\.5 days on eight GPUs", "numeric"),
    ("q03", "How many identical layers is the Transformer encoder stack composed of?",
     "6", A, 3, r"The encoder is composed of a stack of N = 6 identical layers", "numeric"),
    ("q04", "How many parallel attention heads does the Transformer use, and what is the dimension of each head?",
     "8 heads, dk = dv = 64", A, 5, r"we employ h = 8 parallel attention layers, or heads\. For each of these we use dk = dv = dmodel/h = 64", "numeric"),
    ("q05", "What is the inner-layer dimensionality of the position-wise feed-forward network?",
     "2048", A, 5, r"the inner-layer has dimensionality dff = 2048", "numeric"),
    ("q06", "How many warmup steps were used when training the Transformer?",
     "4000", A, 7, r"We used warmup_steps = 4000", "numeric"),
    ("q07", "How many sentence pairs were in the WMT 2014 English-German training dataset?",
     "about 4.5 million sentence pairs", A, 7, r"WMT 2014 English-German dataset consisting of about 4\.5 million sentence pairs", "numeric"),
    ("q08", "What optimizer settings (beta1, beta2, epsilon) were used to train the Transformer?",
     "beta1 = 0.9, beta2 = 0.98, epsilon = 10^-9", A, 7, r"We used the Adam optimizer \[20\] with β1 = 0\.9, β2 = 0\.98 and ε = 10−9", "numeric"),

    # ---------------- BERT (8) ----------------
    ("q09", "What does the acronym BERT stand for?",
     "Bidirectional Encoder Representations from Transformers", B, 1, r"BERT, which stands for Bidirectional Encoder Representations from Transformers", "definition"),
    ("q10", "What GLUE score did BERT report in its abstract, and by how many points?",
     "80.5% (7.7% point absolute improvement)", B, 1, r"pushing the GLUE score to 80\.5% \(7\.7% point absolute improvement\)", "numeric"),
    ("q11", "How many total parameters does BERTLARGE have?",
     "340M", B, 3, r"BERTLARGE \(L=24, H=1024, A=16, Total Parameters=340M\)", "numeric"),
    ("q12", "What percentage of WordPiece tokens does BERT mask during masked language model pre-training?",
     "15%", B, 4, r"we mask 15% of all WordPiece tokens in each sequence at random", "numeric"),
    ("q13", "Which two corpora were used to pre-train BERT, and how many words does each contain?",
     "BooksCorpus (800M words) and English Wikipedia (2,500M words)", B, 5,
     r"we use the BooksCorpus \(800M words\) \(Zhu et al\., 2015\) and English Wikipedia \(2,500M words\)", "list"),
    ("q14", "What Test F1 score did BERT achieve on SQuAD v1.1?",
     "93.2", B, 1, r"SQuAD v1\.1 question answering Test F1 to 93\.2", "numeric"),
    ("q15", "How many Cloud TPU chips were used to train BERTLARGE, and how long did each pre-training run take?",
     "64 TPU chips, 4 days", B, 13, r"BERTLARGE was performed on 16 Cloud TPUs \(64 TPU chips total\)\. Each pretraining took 4 days", "numeric"),
    ("q16", "Besides the masked language model, what second pre-training task does BERT use?",
     "next sentence prediction", B, 2, r"we also use a “next sentence prediction” task that jointly pretrains text-pair representations", "definition"),

    # ---------------- RAG (8) ----------------
    ("q17", "Which pre-trained seq2seq model does RAG use as its generator, and how many parameters does it have?",
     "BART-large, 400M parameters", R, 3, r"We use BART-large \[32\], a pre-trained seq2seq transformer \[58\] with 400M parameters", "numeric"),
    ("q18", "Which retriever does RAG build on?",
     "DPR (Dense Passage Retriever)", R, 2, r"retriever \(Dense Passage Retriever \[26\], henceforth DPR\)", "definition"),
    ("q19", "What Wikipedia dump does RAG use as its non-parametric knowledge source, and how is it chunked?",
     "December 2018 dump, split into disjoint 100-word chunks (21M documents)", R, 4,
     r"we use the December 2018 dump\. Each Wikipedia article is split into disjoint 100-word chunks, to make a total of 21M documents", "list"),
    ("q20", "What is the key difference between the RAG-Sequence and RAG-Token models?",
     "RAG-Sequence uses the same retrieved document for the whole output sequence; RAG-Token can use a different document per token", R, 3,
     r"The RAG-Sequence model uses the same retrieved document to generate the complete sequence", "definition"),
    ("q21", "What Exact Match score does RAG-Sequence achieve on Natural Questions according to the generation/classification results table?",
     "44.5", R, 6, r"RAG-Seq\. 44\.5 56\.8/68\.0", "numeric"),
    ("q22", "How close does RAG get to state-of-the-art pipeline models on the FEVER fact verification task?",
     "within 4.3%", R, 2, r"we achieve results within 4\.3% of state-of-the-art pipeline models", "numeric"),
    ("q23", "On how many open-domain QA tasks does the paper claim state-of-the-art results?",
     "three", R, 1, r"set the state of the art on three open domain QA tasks", "numeric"),
    ("q24", "What search algorithm/library does RAG use to build its document index?",
     "FAISS MIPS index with a Hierarchical Navigable Small World approximation", R, 4,
     r"build a single MIPS index using FAISS \[23\] with a Hierarchical Navigable Small World approximation", "definition"),

    # ---------------- LoRA (8) ----------------
    ("q25", "By how much can LoRA reduce the number of trainable parameters and the GPU memory requirement compared to full fine-tuning of GPT-3 175B with Adam?",
     "10,000 times fewer trainable parameters and 3 times less GPU memory", L, 1,
     r"LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times", "numeric"),
    ("q26", "Unlike adapter methods, what inference characteristic does LoRA claim to have?",
     "no additional inference latency", L, 1, r"unlike adapters, no additional inference latency", "definition"),
    ("q27", "How does LoRA reparameterize a pre-trained weight matrix W0?",
     "W0 + BA with low-rank factors B and A where rank r is much less than min(d, k)", L, 4,
     r"low-rank decomposition W0 \+ ∆W = W0 \+ BA, where B ∈Rd×r, A ∈Rr×k, and the rank r ≪min\(d, k\)", "definition"),
    ("q28", "How are the LoRA matrices A and B initialized at the start of training?",
     "A is initialized with random Gaussian values, B with zeros", L, 4,
     r"We use a random Gaussian initialization for A and zero for B, so ∆W = BA is zero at the beginning of training", "definition"),
    ("q29", "How much VRAM does training GPT-3 175B require with LoRA versus full fine-tuning?",
     "350GB with LoRA versus 1.2TB", L, 5, r"we reduce the VRAM consumption during training from 1\.2TB to 350GB", "numeric"),
    ("q30", "With r = 4 adapting only query and value projections, what does the GPT-3 checkpoint shrink to?",
     "35MB (from 350GB)", L, 5, r"checkpoint size is reduced by roughly 10,000× \(from 350GB to 35MB\)", "numeric"),
    ("q31", "What training speedup does LoRA observe on GPT-3 175B compared to full fine-tuning?",
     "25% speedup", L, 5, r"observe a 25% speedup during training on GPT-3 175B compared to full fine-tuning", "numeric"),
    ("q32", "Which weight matrices does LoRA adapt in most of its experiments?",
     "Wq and Wv (query and value projections)", L, 6, r"we only apply LoRA to Wq and Wv in most experiments", "definition"),

    # ---------------- Deep Residual Learning (8) ----------------
    ("q33", "What top-5 error did the residual net ensemble achieve on the ImageNet test set, and what place did it win?",
     "3.57% error, 1st place in ILSVRC 2015", D, 1,
     r"An ensemble of these residual nets achieves 3\.57% error on the ImageNet test set\. This result won the 1st place on the ILSVRC 2015 classification task", "numeric"),
    ("q34", "How deep were the residual nets evaluated on ImageNet, and how does that compare to VGG?",
     "up to 152 layers, 8x deeper than VGG", D, 1, r"residual nets with a depth of up to 152 layers—8× deeper than VGG nets", "numeric"),
    ("q35", "How is the residual mapping F(x) defined in terms of the desired underlying mapping H(x)?",
     "F(x) := H(x) - x", D, 2, r"fit another mapping of F\(x\) := H\(x\)−x", "definition"),
    ("q36", "What relative improvement on the COCO object detection dataset is attributed solely to the depth of the representations?",
     "28% relative improvement", D, 1, r"we obtain a 28% relative improvement on the COCO object detection dataset", "numeric"),
    ("q37", "What CIFAR-10 classification error does ResNet-110 achieve?",
     "6.43%", D, 7, r"ResNet 110 1\.7M 6\.43", "numeric"),
    ("q38", "How many parameters does the 1202-layer network have and what error does it reach on CIFAR-10?",
     "19.4M parameters, 7.93% error", D, 7, r"ResNet 1202 19\.4M 7\.93", "numeric"),
    ("q39", "Do identity shortcut connections add parameters or computational complexity?",
     "No — they add neither extra parameter nor computational complexity", D, 2,
     r"Identity shortcut connections add neither extra parameter nor computational complexity", "definition"),
    ("q40", "What is the building block equation of a residual unit in the paper?",
     "y = F(x, {Wi}) + x", D, 3, r"a building block defined as: y = F\(x, \{Wi\}\) \+ x", "definition"),
]

# Hand-written paraphrases: used verbatim in the cache pass (semantic-cache test).
PARAPHRASES = {
    "q01": "On English-to-German WMT 2014 translation, what BLEU did the big Transformer model score?",
    "q02": "Training the 41.8 BLEU English-to-French model took how long and how many GPUs were used?",
    "q03": "What is the layer count of the Transformer's encoder stack?",
    "q04": "How many attention heads run in parallel in the Transformer and how wide is each one?",
    "q05": "How wide is the hidden layer inside the Transformer's feed-forward block?",
    "q06": "What warmup step count did the authors pick for training?",
    "q07": "Roughly how many training sentence pairs were in the English-German WMT 2014 set?",
    "q08": "Which Adam hyperparameters were configured when training the Transformer?",
    "q09": "Expand the abbreviation BERT.",
    "q10": "What headline GLUE number did the BERT paper report and how big was the jump?",
    "q11": "What is the parameter count of the large BERT variant?",
    "q12": "During BERT's MLM pre-training, what fraction of tokens gets masked?",
    "q13": "Which datasets fed BERT pre-training and how big is each one?",
    "q14": "What SQuAD v1.1 test F1 did BERT reach?",
    "q15": "How many TPU chips trained BERTLARGE and how many days did it take?",
    "q16": "What auxiliary pre-training objective does BERT use alongside masked LM?",
    "q17": "Which generator backbone powers RAG and how many parameters does it carry?",
    "q18": "RAG relies on which dense retrieval model?",
    "q19": "Which Wikipedia snapshot backs RAG and what chunk size does it use?",
    "q20": "How do RAG-Sequence and RAG-Token differ in how they use retrieved passages?",
    "q21": "What NQ Exact Match did RAG-Sequence report in the results table?",
    "q22": "On FEVER, how near does RAG come to the best pipeline systems?",
    "q23": "For how many open-domain QA benchmarks does the RAG paper claim the SOTA?",
    "q24": "What vector search library indexes RAG's document store?",
    "q25": "How far does LoRA cut trainable parameters and GPU memory versus Adam fine-tuning of GPT-3 175B?",
    "q26": "What does LoRA say about its inference latency compared to adapters?",
    "q27": "How is a frozen weight matrix rewritten in the LoRA formulation?",
    "q28": "What starting values do LoRA's A and B matrices get?",
    "q29": "Training GPT-3 175B with LoRA needs how much VRAM instead of 1.2TB?",
    "q30": "At rank 4 on just the query and value matrices, how small does the GPT-3 LoRA checkpoint get?",
    "q31": "What percentage of training time did LoRA save on GPT-3 175B?",
    "q32": "Which self-attention weights does LoRA usually attach its low-rank matrices to?",
    "q33": "What error rate did the winning ILSVRC 2015 residual ensemble post on ImageNet?",
    "q34": "How many layers did the deepest ImageNet residual net have relative to VGG?",
    "q35": "Express the residual function using the target mapping H(x).",
    "q36": "How much did depth alone improve COCO detection?",
    "q37": "What test error does the 110-layer residual network report on CIFAR-10?",
    "q38": "What happened with the 1202-layer CIFAR-10 model — size and error?",
    "q39": "Do the shortcut connections in residual networks cost extra parameters or compute?",
    "q40": "Write the equation for a residual building block.",
}


def main() -> int:
    pages_path = os.path.join(EVAL_DIR, "corpus_text", "cleaned_pages.json")
    with open(pages_path) as f:
        pages = json.load(f)

    records = []
    errors = []
    for qid, question, answer, doc, page, pattern, qtype in QUESTIONS:
        text = pages[doc].get(str(page))
        if text is None:
            errors.append(f"{qid}: page {page} missing in {doc}")
            continue
        m = re.search(pattern, text)
        if not m:
            errors.append(f"{qid}: snippet pattern not found in {doc} p{page}: {pattern}")
            continue
        snippet = m.group(0)
        records.append({
            "id": qid,
            "question": question,
            "answer": answer,
            "source": doc,
            "page": page,
            "snippet": snippet,
            "type": qtype,
            "paraphrase": PARAPHRASES[qid],
        })

    if errors:
        print("BUILD FAILED:")
        for e in errors:
            print("  -", e)
        return 1

    out_path = os.path.join(EVAL_DIR, "gold_set.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_doc = {}
    for r in records:
        by_doc[r["source"]] = by_doc.get(r["source"], 0) + 1
    print(f"Wrote {len(records)} questions -> {out_path}")
    print("Per-document:", json.dumps(by_doc, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
