# Ampep_Python
This project is for the Ampep project of using Python

## Training Case

### Parameters
py train.py [feature] [Machine Learning Model] [Cross Validation Method] [Fold] [Trees]

### Example
py train.py CTDD RandomForestClassifier ShuffleSplit 10 800

## Testing Case

### Parameters
py test.py [Fasta Path] [Model Path]

### Example
py test.py './data/input.fasta' './model/RandomForestClassifier_ShuffleSplit_800.pkl'