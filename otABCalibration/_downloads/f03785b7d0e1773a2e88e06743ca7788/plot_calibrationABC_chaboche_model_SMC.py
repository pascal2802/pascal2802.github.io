"""
Calibrate Chaboche Model Calibration with an ABCalibration SMC approach
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# %%
# This example investigates a sequential Monte Carlo ABC calibration using the ABCalibration class with default settings to calibrate the Chaboche model.
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
# Calibrate the model with ABC using a SMC approach
# --------------------------------------------------
# The ABC method calibrate the model by sample conditioning : set the parameters
observedParameterIndices = [0]
toCalibrateParameterIndices = [1, 2, 3]
observedOutputIndices = [0]
doeSize = 500  # Size of the prior MonteCarlo sample
posteriorSampleTargetedSize = 50  # Targegeted size of the posterior conditional sample
n_cpus = 12
numberOfIteration = 4

# %% 
# Launch the SMC loop
for i in range(numberOfIteration):
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
    algo.autoAdjustABCCriteria()
    result = algo.getResult()
    conditionalSample = result.getConditionalSample()
    grid = result.drawObservationsVsPredictions()
    fig = otv.View(grid)
    grid = conditionalSample.drawPosteriorInputDistribution()
    fig = otv.View(grid)
    # For the next iteration, the prior distribution is set as the posterior distribution of the current iteration
    distributionParameters = conditionalSample.getPosteriorInputDistribution()

# %% 
# The optimal parameter set of the last SMC iteration is 
dfCalibration = result.getThetaMAPAsDataFrame()
print(dfCalibration)

