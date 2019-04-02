Python files - No command line arguments required for any 

MissingDataScriptModified.py - run with background.csv in folder. Outputs "outputModified.csv" with imputed median values.

baselineModelPredictions.py - Decomment specified line for desired model. Tests regression with all relevant mother
and father features. 

fatherMotherComparative.py - Analyzes mother and father predictivity at all time points. Plots results in group bar graph.

selectFeatures.py - Produces a dataframe with desired features extracted from outputModified.csv.
Used as an import in other programs, not as itself.

depressionVgritFeatureGenerator.py - Isolates features relevant to depression in a dataframe.
Computes a combined depression metric. (Note: reads from background.csv, not outputModified.csv)

depressionVgrit.py - Computes a correlation between the combined depression metric and grit.
