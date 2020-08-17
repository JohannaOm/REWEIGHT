
# LM4KG: Improving Common Sense Knowledge Graphs with Language Models

This repository contains code for the 2020 ISWC paper 
"LM4KG: Improving Common Sense Knowledge Graphs with Language Models"

## Contents
We provide code for applying REWEIGHT to the common sense Knowledge Graphs (KGs) ConceptNet, WebChild, and YAGO. 

REWEIGHT weights the triples of common sense KGs based on how reasonable their content is:

![REWEIGHT Pipeline](/imgs/reweight_pipeline.pdf)

REWEIGHT generates sentences from the triples, corrects the sentences with a grammar checker, feeds the sentences to a language model, 
and converts the perplexities (i.e. how likely a sentence is to occur according to the language model) 
into edge weight for the graph. 

Since grammar checker and language model can be freely chosen, this repository contains the code for sentence generation, 
perplexity to score transformation, and feeding back scores to the graph.

## Usage
Running REWEIGHT on a KG requires the following steps:
* Download the KG you wish to REWEIGHT
* Transform the graph to ConceptNet format 
(**Note:** Since KGs all have different formats, this requires some manual effort. 
Examples on transforming WebChild and YAGO are available under 
`sentence_construction/WebChild_to_sentence.py` and `sentence_construction/yago_to_sentence.py`)
* Transform the graph triples to sentences: `sentence_construction/graph_to_sentence.py`
* (Optional) Run a grammar checker on the sentences (Original paper uses https://github.com/awasthiabhijeet/PIE)
* Generate perplexities for each sentence through a language model 
(Original paper uses https://github.com/xu-song/bert-as-language-model)
* Transform the perplexities to edge scores and feed them back into the graph: 
`graph_reweighting/perplexities_to_scores.py`

## Downloads
The following links can be used to download the weighted KGs and 
KG enriched embeddings presented in the paper:
* Weighted Knowledge Graphs
    * ConceptNet REWEIGHT: *Link will be provided soon.*
    * ConceptNet REWEIGHT<sub>light</sub>: *Link will be provided soon.*
* Knowledge Graph enriched word embeddings (through retrofitting):
    * ConceptNet NumBERTBatch: *Link will be provided soon.*
    * ConceptNet NumBERTBatch<sub>light</sub>: *Link will be provided soon.*
    
