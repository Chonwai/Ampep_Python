# Ampep_Python
This project is for the Ampep project of using Python

## Cross Validation Testing Case

### Parameters
py train.py [Feature] [Machine Learning Model] [Cross Validation Method] [Fold] [Trees] [Step]

### Example
py train.py CTDD RandomForestClassifier ShuffleSplit 10 800 30

#### Feature
The feature method is base on the iFeature. For example in this program, we use 'CTDD' feature.

#### Machine Learning Model
We support six training model and the default model is RandomForestClassifier.
* RandomForestClassifier
* BaggingClassifier
* ExtraTreesClassifier
* RandomTreesEmbedding
* AdaBoostClassifier
* GradientBoostingClassifier

#### Cross Validation Method
There are four cross-validation methods and the default method is ShuffleSplit.
* ShuffleSplit
* StratifiedKFold
* StratifiedShuffleSplit
* RepeatedStratifiedKFold

#### Fold
This parameter is for the cross-validation fold and the default value is 10.

#### Trees
Amount of trees. The default trees are 100.

#### Step
The number of looping for the training. Each loop will increases 100 trees. The default of step is 30.

## Training Case

### Parameters
py train.py [Feature] [Machine Learning Model] [Trees]

### Example
py train.py CTDD RandomForestClassifier 800

#### Feature
The feature method is base on the iFeature. For example in this program, we use 'CTDD' feature.

#### Machine Learning Model
We support six training model and the default model is RandomForestClassifier.
* RandomForestClassifier
* BaggingClassifier
* ExtraTreesClassifier
* RandomTreesEmbedding
* AdaBoostClassifier
* GradientBoostingClassifier

#### Trees
Amount of trees. The default trees are 100.

## Testing Case

### Parameters
py test.py [Feature] [Fasta Path] [Model Path]

### Example
py test.py CTDD './data/input.fasta' './model/RandomForestClassifier_800.pkl'

#### Feature
The feature method is base on the iFeature. For example in this program, we use 'CTDD' feature.

#### Fasta Path
The path of the input .fasta file.

#### Model Path
The path of the model file.