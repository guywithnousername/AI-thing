# Credits

My dataset came from a [dataset](https://www.kaggle.com/datasets/sujithmandala/simple-loan-classification-dataset) from [kaggle.com](kaggle.com) 

# Installed Modules

This project uses a virtual environment. To create it, do
`python3 -m venv .venv`.
`.venv` can be any name.

To install something, first activate the virtual environment by writing 
`. .venv/bin/activate`.
After this, do `pip install [module]`. 

For example, if the list says to install `pandas`, do `pip install pandas`.

The list is accessible in dependencies.txt, was generated by doing `pip freeze > dependencies.txt`. 

# Notes

These are mostly just things I learned. I took most of this knowledge from [kaggle.com](kaggle.com). 

Dot notation is used to select the prediction target. The column predicted is usually a variable called `y`. 
Features are the inputs that are inputted into the model. Those features are reffered to as a variable called `X`.
What I am currently using is a model called a decision tree regressor. 