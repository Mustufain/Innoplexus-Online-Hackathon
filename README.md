# Innoplexus-Online-Hackathon
2 Day online hackathon


Researchers often build upon the work of other researchers and as a result cite their research articles for inspiration or comparison of work.


But for some publication firms, it is a problem when a research article cites irrelevant articles as references.


So, in this competition, a publication firm has presented you with a problem of identifying the articles that another article will cite as a reference.

For a particular article, the client has provided information such as its abstract, author name, title of the article etc. More details on the information can be found below. Also, it is worth noting that, research articles in both train and test are divided on the basis of set. A research article belonging to say set 1 can only cite articles from set 1 and not any other set.


Training data set has 9 sets in total, whereas Test data set has 9 sets. The Test has been further divided into public and private data set each containing 5 and 4 sets respectively.


Note: All sets irrespective of whether in train, public or private are independent of each other and share no article ids(pmid) in common.

Data Dictionary

 

*information_train.csv, information_test.csv - tab separated files

 **Variable       Meaning

**abstract: Text containing the abstract of the article

**article_title: Title of the article

**author_str: Name of the Authors of the article

**pmid: Id for the article

**pub_date: Publication date of the article/manuscript

**set: The set to which the article belongs

**full_Text: The complete text of the research article



*Train.csv - comma separated file

**Variable Meaning

**pmid: Id for the article

r**ef_list : pmid of the articles that this article has cited



*Test.csv - comma separated file

**Variable Meaning

**pmid: Id for the article


Evaluation Metric

The Evaluation Metric for this competition is f1 weighted by samples. An example of the same is provided below.

Actuals: Both rows belong to same set.

Row1: [1,3]

Row2: [1,4]

 

Predicted:

Row1 : [1,4]

Row2: [2,3]

 

F1_score for Row1 = 0.5

F1_score for Row2 = 0

Average F1_score = (0.5+0)/2 = 0.25

Final F1_score is calculated by taking average of Average F1_score across sets.

 
