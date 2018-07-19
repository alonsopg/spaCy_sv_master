# SpaCy_sv_master
===========================
### 

E-mail: wofmanaf@gmail.com

    Author: wofmanaf.   Email: wofmanaf@gmail.com
    Created for Zhang Heng! 

Swedish language model, using SpaCy, including Vocabulary, syntax, entities, vectors. The pre-trained Model
[sv_model-0.0.0.tar.gz](https://drive.google.com/open?id=1hpiyTTdeT1kG7m7_GCnOZG_j2Tpm8GGg) is available to be download.

### Requirements
* spaCy v2.0.11 
* Python 3.6
* Numpy 
* Pandas 

### Example
--------------------------------
```python
>>> import spacy
>>> nlp = spacy.load('sv_model')
>>> doc = nlp("Självkörande bilar förskjuter försäkringsansvar mot tillverkare.")
```

Note that this project import some source code from others, we will write a reference list soon!
 

