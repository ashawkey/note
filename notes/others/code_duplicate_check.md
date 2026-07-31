# code duplicate check

### nltk

* word_tokenize

  extract English word from a string.

### gensim

* corpus

  a collection of documents (texts, strings).

* bow (bag of words)

  use one-hot encoding of word to represent a document.

* tf-idf (term frequency - inverse document frequency)

  model that change bow representation to a dense vector representation.

  can reflect how important a word is to a document.

* similarities

  compare similarity of vectors.


### example

```python
# check for duplicates
users = list(results.keys())
users.sort()

for problem_id in range(len(problems)):
    # build dictionary
    docs = []
    for user in users:
        if problem_id in results[user]:
            docs.append([word.lower() for word in word_tokenize(results[user][problem_id][1])])
    dictionary = gensim.corpora.Dictionary(docs)
    bows = [dictionary.doc2bow(doc) for doc in docs]
    tfidf = gensim.models.TfidfModel(bows)
    sims = gensim.similarities.Similarity(tempdir + os.sep, tfidf[bows], num_features=len(dictionary))
    # check duplicates
    for user in users:
        if problem_id in results[user]:
            result_type, source = results[user][problem_id]
            query_doc = [word.lower() for word in word_tokenize(source)]
            query_bow = dictionary.doc2bow(query_doc)
            query_tfidf = tfidf[query_bow]

            for similarity, user2 in zip(sims[query_tfidf], users):
                if user2 == user: continue
                if similarity > DUPTHRESH:
                    if VERBOSE:
                        print(f'[INFO] problem {problem_id}: {user} <-- {similarity:.3f} --> {user2}')
                    result_type += f'\n### [DUP] {user2}: similarity = {similarity:.3f}'
                    results[user][problem_id][0] = result_type
```

