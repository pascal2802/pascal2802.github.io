"""
Getting Started with Chaboche Model Calibration: Minimal Setup using ABCalibration
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# %%
# This example demonstrates how to use the ABCalibration class with default settings to calibrate the Chaboche model.
# We recall that the Chaboche model is the following :
#
# .. math::
#   \sigma = G(\epsilon,R,C,\gamma) = R + \frac{C}{\gamma} (1-\exp(-\gamma\epsilon))
#
# where:
#
# - :math:`\epsilon` is the strain,
# - :math:`\sigma` is the stress (Pa),
# - :math:`R`, :math:`C`, :math:`\gamma` are the parameters.


# %%
# | Loading python modules

# %%
import openturns as ot
from openturns.usecases import chaboche_model
import math
import imp
import otABCalibration as otABC
import openturns.viewer as otv
import matplotlib.pyplot as plt
import openturns.viewer as otv
import pickle

imp.reload(otABC)

# ot.Log.Show(ot.Log.NONE)

# %%
# Define the Observations
# ==================================================
# In practice, we generally use a data set which has been obtained from
# measurements.
# This data set can be loaded using e.g. :meth:`~openturns.Sample.ImportFromCSVFile`.
# Here we import the data from the
# :class:`~openturns.usecases.chaboche_model.ChabocheModel`
# class.
cm = chaboche_model.ChabocheModel()
print(cm.data)
observedParameters = cm.data[:, 0]  # Strain
observedVariables = cm.data[:, 1]  # Stress
numberOfObservations = cm.data.getSize()

# %%
# Define the prior joint distribution of the parameter to calibrate :math:`\pi(\theta)`
Rdistribution = ot.Uniform(500.0e6, 800.0e6)
Cdistribution = ot.Uniform(1000.0e6, 7000.0e6)
gammaDistribution = ot.Uniform(1.0, 15.0)
distributionParameters = ot.ComposedDistribution(
    [Rdistribution, Cdistribution, gammaDistribution]
)
distributionParameters.setDescription(["R", "C", r"$\gamma$"])

# %%
# Calibrate the model with ABC
# --------------------------------------------------
# The ABC method calibrate the model by sample conditioning
observedParameterIndices = [0]
toCalibrateParameterIndices = [1, 2, 3]
observedOutputIndices = [0]
doeSize = 15000  # Size of the prior MonteCarlo sample
posteriorSampleTargetedSize = 300  # Targegeted size of the posterior conditional sample
n_cpus = 12
algo = otABC.ABCalibration(
    cm.model,
    observedParameterIndices,
    toCalibrateParameterIndices,
    observedOutputIndices,
    observedParameters,
    observedVariables,
    distributionParameters,
    doeSize,
    posteriorSampleTargetedSize,
    n_cpus=n_cpus,
)
algo.setABCCriteriaDescription([r"$CvRMSE_{\sigma}$", r"$NMBE_{\sigma}$"])
algo.run()

# %%
# Investigate the results
result = algo.getResult()

# %%
# draw posterior input distribution to analyse calibration
# it can be seen that :math:`\gamma` cannot be idenfied accurately but that some correlation with
# the two other parameters are present.
# The Default ABC criteria is mentionned on the title graph and are:
#
# - :math:`CvRMSE < 0.2` for all observed variables
# - :math:`0.05 < NMBE < 0.05` for all observed variables
conditionalSample = result.getConditionalSample()
grid = conditionalSample.drawPosteriorInputDistribution()
fig = otv.View(grid)

# %%
# on the next picture, the residuals distribution of the computed optimal point (the point that maximise the posterior input distribution infered from the empiric posterior sample) is analysed.
# the figure suggets that the discrepencies between model prediction and observed output are mostly due to measurment erros as the residuals are gaussian and centered.
print(result.getParameterMAP())
grid = result.drawResiduals()
fig = otv.View(grid)
grid = result.drawObservationsVsPredictions()
fig = otv.View(grid)
grid = conditionalSample.drawPosteriorMarginalOutputDistribution()
fig = otv.View(grid)

# %%
# Previous graph suggest that criteria selection are too large and can be adjusted
criteria = algo.autoAdjustABCCriteria()

# %%
# Re-generate the results
result = algo.getResult()
conditionalSample = result.getConditionalSample()
print(result.getParameterMAP())
grid = conditionalSample.drawPosteriorInputDistribution()
fig = otv.View(grid)
grid = result.drawResiduals()
fig = otv.View(grid)
grid = result.drawObservationsVsPredictions()
fig = otv.View(grid)
grid = conditionalSample.drawPosteriorMarginalOutputDistribution()
fig = otv.View(grid)

# %%
# Display the calibration results in a dataframe, including confidence intervals for the parameters
dfCalibration = result.getThetaMAPAsDataFrame()
print(dfCalibration)


# %%
# Cross validate the calibration
# ++++++++++++++++++++++++++++++++++++++++++++++++++
# The objective is to perform cross-validation of the calibration.
# The global design of experiment is evaluated with the provided model,
# and the ABC criteria are computed using only the observations provided within the indexTrain list.
# This approach allows for the validation of the model's predictive performance on observations
# that were not used during the calibration process.
# The cross-validation does not require additional model evaluations, making it an efficient method
# for assessing the model's generalization capabilities.
indexTrain = list(range(numberOfObservations))
validationIndice = 0
indexTrain.pop(
    validationIndice
)  # Remove the first obervation from the train set to use it as validation point
resultValidation = algo.crossValidationABC(indexTrain)
grid = algo.drawCrossValidationObvservationsVsPrediciton(
    resultValidation, validationIndice
)
grid.setTitle(f"Obsersation used as validation : {validationIndice}")
fig = otv.View(grid)
graph = resultValidation.drawObservationsVsPredictions()
fig = otv.View(graph)

# %%
# Display statistics for model function calls
# CacheHits represents the number of calls that were served from the cache
print(algo.model.getCallsNumber())
print(algo.model.getCacheHits())
print(algo.model.getCacheInput().getSize())

# %%
# Export
# ++++++++++++++++++++++++++++++++++++++++++++++++++
# Export the model for futur Reuse
algo.exportCalibrationWithPickle("algo.pkl")
with open("algo.pkl", "rb") as f:
    calibration = pickle.load(f)
# Test another criteria
calibration.setABCCriteria(ot.Interval([0, -0.02], [0.1, 0.02]))
# Cache Size before running again the run method
print(calibration.model.getCacheInput().getSize())
calibration.run()  # No computational cost thanks to the Cache mechanism of the memoize function
result = calibration.getResult()
print(result.getParameterMAP())
# No additional model function evaluation
print(calibration.model.getCacheInput().getSize())
