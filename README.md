# lsi-tagger  
LSI Tagger is a creatively named package that, as one might assume, extracts keywords/tags using Latent Semantic Indexing (LSI). Developed in an effort to both improve user experience and provide interpretability during product search in an e-commerce setting, it can be used as a content recommendation system (given input text find the most similar candidate products), an explainer for the algorithm that's serving product recommendations based on similarity, a filter to help the user fine-tune their search, and/or suggest keywords/tags that may help the user tweak their search.  
  
Compared to other keyword extraction methods, this package does so on a pairwise basis so that keywords/tags for a given document 1) may change given new candidate documents and 2) are much more likely to be able to explain the relationship to its candidates.  
  
In fewer words, this package's main capabilities are:  
1) Explain search results and allow for targeted re-ranking: Extract keywords from an input and N candidate documents, and re-rank.
2) Suggest tweaks to a user's current search based on the trained corpus: Extract "adjacent" keywords given extracted input tags.
  
  
## Installation  
### Dependencies  
lsi-tagger requires:  
Python (>= 3.8.10)  
NumPy (>= 1.22.3)  
Gensim (>= 4.1.2)  
nltk (>= 3.7)  
  
### User installation
`pip install lsi-tagger`
