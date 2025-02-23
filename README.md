# Mobile-Computing
This Smart Home project was completed for the Mobile Computing course CSE535 through ASU

# Notes for project files and functions
imports both frameExtractor from frameextractor and
HandShapeFeatureExtractor from handshape_feature_extractor

extracts all middle frames from traindata
creates png files for all and saves within trainFrames
adds extracted frames into trainingArray

extracts all middle frames from test
creates png files for all and saves within testFrames
adds extracted frames into testArray

uses cosine similarity to compare the trainingArray and testArray
adds the results into Results.csv

Results.csv includes 50 result of the determined similarity of the arrays.
