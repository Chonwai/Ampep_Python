# Ampep_Python
This project is for the Ampep project of using Python

## Cross Validation Testing Case

### Parameters
py train.py [Feature] [Machine Learning Model] [Cross Validation Method] [Fold] [Trees] [Step]

### Example
py train.py CTDD RandomForestClassifier ShuffleSplit 10 800 30

## Training Case

### Parameters
py train.py [Feature] [Machine Learning Model] [Trees]

### Example
py train.py CTDD RandomForestClassifier 800

## Testing Case

### Parameters
py test.py [Feature] [Fasta Path] [Model Path]

### Example
py test.py CTDD './data/input.fasta' './model/RandomForestClassifier_800.pkl'