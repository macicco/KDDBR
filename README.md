# Competition: Can I make a wish? Predicting the presence of meteors in images
------------------------------------------------------------------------------

Here I present the implementation code of my team's approach for solving the 1st KDD-BR (Brazilian Knowledge Discovery in Databases) competition hosted on Kaggle.

##  &#42;&#42;&#42;Warning: we have not yet added the code to the repository, but we will do so after the conference dinner, at October, 4th.&#42;&#42;&#42;


## Requirements
Python 3.5 with the following libraries:

* [Numpy](http://www.numpy.org/) (>=1.13.1)
* [Scikit-learn](http://scikit-learn.org/) (>=0.19.0)
* [Imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) (>=0.3.0)

## Usage
```python main.py [options] [outputFile]``` 

```
outputFile: 
   path to the output file  

Options:

   -p polynomial_features : (default 0)
       0 -- false (do not generate polynomial features) 
       1 -- true (generate polynomial features) 
``` 

We have selected two final submissions for judging. 

To replicate the first selected submission (the one that did not generated polynomial features), use:

		
		python main.py outputFile.csv

To replicate the second selected submission (the one that generated polynomial features), use:

		
		python main.py -p 1 outputFile.csv