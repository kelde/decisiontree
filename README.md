decisiontree
============

Background
----------

One of the machine learning methods used in predictive statistics is the “learning decision tree”. A decision tree is used to predict an outcome from a set of observations based on the set of values present in each observation. A decision tree is much like that little pinball-like machine you may have seen on “The Price is Right”, however, while the path of the little disk in that show was entirely random, the path formed by a decision tree is not random, it is formed through an application of statistical probability. A decision tree “learns” the path for each outcome variable from the set of data that it is built from.

In order to understand how this works, one needs to understand two concepts in information science. These concepts are information gain, and entropy.

Entropy is essentially what it sounds like… randomness. When we apply the concept of entropy to information, or data, we are talking about the organization of the data. More specifically, entropy is a way of quantifying the likelihood that one will select the correct value of a random variable, in other words, entropy is the probability that a given outcome will be true in a set of data. A set of data has low entropy if the chance of selecting a random variable in that set is particularly high.

For example, imagine that we have a set of 100 students and we are interesting in selecting a student who majors in political science. Imagine that 50% of the students major in science and 50% of the students major in the humanities. Now imagine that 50% of the students who major in humanities major in political science. The percentage of students who major in political science is therefore 25%. The chance that we will select a student who majors in political science from the total student body is pretty low (25%). The entropy of the entire set of students, therefore, is fairly high. However, if we limit the set of students to just students who study humanities, we can increase our chance of selecting a student who majors in political science significantly (to 50%). We thereby reduce the entropy of the overall set by limiting the set of students to just the students who major in humanities. If you were to repeat this process over and over again, you might begin to see how a decision tree eventually leads us to a prediction, or classification, of an observation, based on the different “splits” that occurred in order to reduce entropy.

The second concept to understand is the concept of information gain. If we think of entropy as the likelihood that we will select the correct value of a random variable, then information gain is the increase in likelihood that we will select the correct value of a random variable following some type of action taken on the data. You can think of information gain as a quantification of the decrease in entropy caused by taking some action on the data. For example, when we reduced the set of students to just the students who majored in humanity, we increased the odds that we would select a student who majored in political science. We therefore “gained information” by reducing the entropy of the set of data. We use the concept of information gain in a decision tree to select the “node”, or the attribute, that we use to make each split in the tree.

To form a decision tree, these two concepts are recursively applied to the set of data. A decision tree recursively reduces the entropy in any given set of observations until it has eventually decreased the entropy to the point where one can be guaranteed to select the proper classification based on the values in the observation. This guarantee (i.e. perfectly “fitting” the data) leads to one of the key problems of decision trees, over-fitting, which can be addressed by reducing the specificity of the tree through a practice called pruning.

For more information on decision trees, particularly the algorithm that we will explore here (called ID3), you can refer to the following link:

http://www.doc.ic.ac.uk/~sgc/teaching/pre2012/v231/lecture11.html

This link does a great job of explaining the ID3 algorithm and is written by Simon Colton of the University of London.

There are several different implementations of decision tree algorithms. Searching will turn up algorithms in R and python’s scikit-learn. While these algorithms are obviously of very high quality, using them does not necessarily teach us how the inner workings of a decision tree operate. For example, the calculations used to determine entropy and information gain, as well as each node in the decision tree are all hidden behind the scenes. To learn more about the algorithm, I wrote a python class that allows me to iteratively build a decision tree. The class does not implement the recursive element of the algorithm (I hope to update it at a later point to include this functionality), however, it does allow us to think through the process of building a decision tree, including entropy and information gain.

I’ve transcribed the algorithm from Simon Colton’s website below:

1) Choose the attribute of the full set which has the highest information gain, and set this as the root node
2) For each value that the attribute can take, draw a branch from the root node
3) For each branch, calculate the set of observations for that branch (i.e. limiting the set to just values that satisfy the criterion of the branch
3A) If the set is empty, choose the default category, which will be the majority of observations from the total set
3B) If the set only contains observations from a single category, then set a leaf node and make the value of that leaf node the category
3C) Otherwise, remove the attribute from the set of attributes that are candidates to be a new node. Then determine which attribute scores the highest on information gain with respect to the set calculated in step 3. The new node will then start the cycle again from step 2.

While a full blown implementation of this algorithm would implement the recursion to fully build out the tree, I’ve opted to take a simpler approach and simply implement several methods that allow me to step through the algorithm in “interactive mode”, so to speak.

I’ve decided to use a dataset from UCI’s machine learning repository. The dataset contains data on 470 surgeries that occurred on patients suffering from lung cancer. The outcome variable is whether or not the patient dies within one year of receiving the surgery (‘Risk1Y’) and the variables are various attributes of the patient and their illness, including age, diagnosis, tumor size, the outcomes of various pulmonary function tests, whether the patient smokes, whether they experienced a cough before surgery, and more. The full dataset can be found here:

https://archive.ics.uci.edu/ml/datasets/Thoracic+Surgery+Data

The Algorithm:  Step-by-step using the DecisionTree class
---------------------------------------------------------

The first step in the algorithm is to calculate the entropy of the entire set of data with respect to the outcome variable (‘Risk1Y’). Since we are using a binary classification (the patient either died or lived), we can use the formula to calculate entropy for a binary classification:

Entropy(s)=−p+log2(p+)−p−log2(p−)
Where p+ is the proportion of outcomes for which the value is positive (i.e. the classification is true) and p− is the proportion of outcomes for which the value is negative (i.e. the classification is false).

To do this using the python class:

```python
learningTree = DecisionTree("/Users/kelder86/decision_tree/thoracic_surgery.csv", "Risk1Y")
learningTree.setFieldNames(['Diagnosis', 'FVC', 'FEV1', 'PerformanceStatus', 'Pain', 'Haemoptyisis', 'Dyspnoea', 'Cough', 'Weakness', 'SizeOfTumor', 'Diabetes', 'MI', 'PAD', 'Smokes', 'Asthma', 'Age', 'Risk1Y'])
learningTree.loadFile()
learningTree.printExcludedFields()
learningTree.printSetConditions()
learningTree.setCurrentSet()
learningTree.calculateSetEntropy()
learningTree.printEntropy()
```

This gives us the following output:

Positive outcomes in given set: 70.0
Negative outcomes in given set: 400.0
Entropy of given set is: 0.607171654871

You can see that the formula for entropy in this case would be:

Entropy(s)=−(70/470)log2(70/470)−(400/470)log2(400/470)=0.607171654871

Now we need to cycle through each one of the attributes, except for the outcome variable, and determine which one of the attributes will reduce entropy by the greatest amount. In other words, which one of the attributes gives us the greatest information gain.

To calculate information gain for a binary classification problem, we use the following calculation:

Gain(S,A)=Entropy(S)−∑v∈Values(v)(Sv/S)Entropy(Sv)
To do this with the class, we do the following:

```python
learningTree.getHighestGain()
learningTree.printMaxGain()
```

This produces the following output:

Attribute with maximum gain is: {‘name’: ‘Diagnosis’, ‘gain’: 0.024284175235364205}

Remember, at this point we are processing the entire set of data (all 470 records) and we are not filtering the data in any way. The output above suggests that the diagnosis field will reduce the entropy of the entire set of data by the greatest amount out of all of the available attributes in the file.

Let’s get a little more output to see how this works. Below is the additional output from the getHighestGain() method for the first attribute processed:

| Output |
| ------ |
|Total records in set: 470.0|
|Field is: Diagnosis|
|Total number with value DGN1: 1; Total number with value DGN1 and outcome F: 1|
|Entropy of the set with value: DGN1: 0.0|
|Total number with value DGN2: 52; Total number with value DGN2 and outcome F: 40; Total number with value DGN2 and outcome T: 12|
|Entropy of the set with value: DGN2: 0.779349837292|
|Total number with value DGN3: 349; Total number with value DGN3 and outcome F: 306; Total number with value DGN3 and outcome T: 43|
|Entropy of the set with value: DGN3: 0.538515706679|
|Total number with value DGN4: 47; Total number with value DGN4 and outcome F: 40; Total number with value DGN4 and outcome T: 7|
|Entropy of the set with value: DGN4: 0.607171654871|
|Total number with value DGN5: 15; Total number with value DGN5 and outcome F: 8; Total number with value DGN5 and outcome T: 7|
|Entropy of the set with value: DGN5: 0.996791631982|
|Total number with value DGN6: 4; Total number with value DGN6 and outcome F: 4|
|Entropy of the set with value: DGN6: 0.0|
|Total number with value DGN8: 2; Total number with value DGN8 and outcome F: 1; Total number with value DGN8 and outcome T: 1|
|Entropy of the set with value: DGN8: 1.0|
|Gain for this attribute: 0.0242841752354|

We can see from this (knowing the calculation for gain) that the final formula for information gain for this attribute would be:

Gain(S,A)=0.607171654871−(1/470)(0)−(52/470)(0.779349837292)−(349/470)(0.538515706679)−(47/470)(0.607171654871)−(15/470)(0.996791631982)−(4/470)(0)−(2/470)(1.0)=0.0242841752354

Now we find the attribute with the maximum gain:

| Output |
| ------ |
|Attribute is: Diagnosis; Gain for this attribute: 0.0242841752354|
|Attribute is: FVC; Gain for this attribute: 0.0203096548951|
|Attribute is: FEV1; Gain for this attribute: 0.00305944453271|
|Attribute is: PerformanceStatus; Gain for this attribute: 0.00652051279898|
|Attribute is: Pain; Gain for this attribute: 0.00212629826488|
|Attribute is: Haemoptyisis; Gain for this attribute: 0.00289481096538|
|Attribute is: Dyspnoea; Gain for this attribute: 0.0067109933434|
|Attribute is: Cough; Gain for this attribute: 0.00603793299662|
|Attribute is: Weakness; Gain for this attribute: 0.00495240811225|
|Attribute is: SizeOfTumor; Gain for this attribute: 0.0209567260041|
|Attribute is: Diabetes; Gain for this attribute: 0.00720382164832|
|Attribute is: MI; Gain for this attribute: 0.000992338695742|
|Attribute is: PAD; Gain for this attribute: 0.000868693043524|
|Attribute is: Smokes; Gain for this attribute: 0.00600290579821|
|Attribute is: Asthma; Gain for this attribute: 0.000992338695742|
|Attribute is: Age; Gain for this attribute: 0.0036530835234|
|Attribute with maximum gain is: {‘name’: ‘Diagnosis’, ‘gain’: 0.024284175235364205}|

Out of all of the available attributes, the maximum gain is for the Diagnosis variable. We therefore set this attribute as the root node and we get all of the possible values for this attribute.

```python
learningTree.setNodeBranches('Diagnosis')
learningTree.printBranches()
['DGN8', 'DGN1', 'DGN3', 'DGN2', 'DGN5', 'DGN4', 'DGN6']
```

These are the possible values of diagnosis. Now in a fully recursive implementation, the process would then cycle through each one of these branches, setting a node for each branch, and performing all of the calculations until the tree is exhausted. In our case we are going to manually traverse just one more level down for explanation.

We first need to calculate the new set for the branch. We can do this by excluding the values that we have already used by traversing the branch from the total set. For example, to calculate Sv for the DGN3 branch:

```python
learningTree.setSetConditions([{ 'name' : 'Diagnosis', 'value' : 'DGN3' }])
learningTree.setExcludedFields(['Risk1Y', 'Diagnosis'])
learningTree.processBranch('DGN3')
learningTree.printEntropy()
learningTree.printMaxGain()
```

The conditions will filter the set to only contain observations where ‘Diagnosis’ is equal to ‘DGN3′. As we traverse down the tree we continue to add these conditions to represent the path that we took down to the leaf node. In the case below, the next node following the ‘Diagnosis’ node down branch ‘DGN3′ is the ‘SizeOfTumor’ attribute.

| Output |
| ------ |
|Calculating nodes for branch: DGN3|
|Fields currently excluded as potential nodes: ['Risk1Y', 'Diagnosis']|
|Conditions placed on current set:|
|[{'name': 'Diagnosis', 'value': 'DGN3'}]|
|Attribute is: FVC	Gain for this attribute: 0.0178882581378|
|Attribute is: FEV1	Gain for this attribute: 0.00440246814813|
|Attribute is: PerformanceStatus	Gain for this attribute: 0.00889217478759|
|Attribute is: Pain	Gain for this attribute: 0.0119114583755|
|Attribute is: Haemoptyisis	Gain for this attribute: 0.00515530395553|
|Attribute is: Dyspnoea	Gain for this attribute: 0.00720521042675|
|Attribute is: Cough	Gain for this attribute: 0.00429029313854|
|Attribute is: Weakness	Gain for this attribute: 0.0032716018855|
|Attribute is: SizeOfTumor	Gain for this attribute: 0.0258570293306|
|Attribute is: Diabetes	Gain for this attribute: 0.00582417248134|
|Attribute is: MI	Gain for this attribute: 0.00109042213585|
|Attribute is: PAD 	Gain for this attribute: 0.00273871968586|
|Attribute is: Smokes	Gain for this attribute: 0.00443986103654|
|Attribute is: Asthma	Gain for this attribute: 0.00109042213585|
|Attribute is: Age	Gain for this attribute: 0.0032465590173|
|Positive outcomes in given set: 43.0|
|Negative outcomes in given set: 306.0|
|Entropy of given set is: 0.538515706679|
|Attribute with maximum gain is: {‘name’: ‘SizeOfTumor’, ‘gain’: 0.025857029330573714}|
