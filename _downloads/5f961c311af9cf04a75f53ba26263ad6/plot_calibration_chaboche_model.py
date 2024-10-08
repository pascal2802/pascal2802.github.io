"""
Calibrate Chaboche model using ABCalibration Class
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

# %%
# This example aims to illustrate the use of the ABCalibration class with the Chaboche model
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
import otABCalibration.ABC_ClassProto as otABCC
import openturns.viewer as otv
import matplotlib.pyplot as plt
import openturns.viewer as otv


imp.reload(otABCC)

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
# Set Calibration prior distribution
# --------------------------------------------------
# The prior observed parameters uncertainty distribution parameter is set
# random uncertainty will be add to the observed parameters sample while evaluated ABC DOE.
StrainUdistribution = ot.Normal(0, 0.001)

# %%
# Define the prior joint distribution of the parameter to calibrate :math:`\pi(\theta)`

Rdistribution = ot.Uniform(500.0e6, 800.0e6)
Cdistribution = ot.Uniform(1000.0e6, 7000.0e6)
gammaDistribution = ot.Uniform(1.0, 15.0)
distributionParameters = ot.ComposedDistribution(
    [Rdistribution, Cdistribution, gammaDistribution]
)
distributionParameters.setDescription(["R", "C", r"$\gamma$"])

thetaPrior = distributionParameters.getMean()
print(thetaPrior)

# %%
# Build a joint distribution between parameters to calibrate prior and observed parameter uncertainty prior

distributionInputs = ot.ComposedDistribution(
    [StrainUdistribution, Rdistribution, Cdistribution, gammaDistribution]
)
distributionInputs.setDescription([r"$U_{\varepsilon}$", "R", "C", r"$\gamma$"])


# %%
# Set the calibration criteria
# ==================================================
# modeller need to define the computation of the criteria to define a calibrated model based on the returned sample by the evaluation of all the observation point


def computeABCCriteria(samplePrediction, observedVariableSample):
    """
    function to compute criteria that will be used for ABC calibration
    In future ABC class calibration, function to be provided by modeller

    Parameters
    ---------
    samplePrediction : :class:`~openturns.Sample`
        Take as input the return sample from the evaluation of _exec function for all the point in the sample of observed parameters for a given candidate point of ParameterToCalibrate



    Returns
    -------
    pointCriteria : :class:`~openturns.Point`

    for a given ParamterToCalibrate Point, return a point with computed criteria (typically CvRMSE, NMBE)
    """

    residuals = samplePrediction - observedVariableSample

    pointCriteria = ot.Point(4)

    # compute RMSE
    RMSE_stress = math.sqrt(residuals.computeRawMoment(2)[0])
    MBE_stress = residuals.computeMean()[0]
    CvRMSE_stress = RMSE_stress / (observedVariableSample).computeMean()[0]
    NMBE_stress = MBE_stress / (observedVariableSample).computeMean()[0]

    pointCriteria[0] = RMSE_stress
    pointCriteria[1] = MBE_stress
    pointCriteria[2] = CvRMSE_stress
    pointCriteria[3] = NMBE_stress

    return pointCriteria


# %%
# test the function with the :math:`\theta_{prior}` computed above
calibratedIndices = [1, 2, 3]
mycf = ot.ParametricFunction(cm.model, calibratedIndices, thetaPrior)
priorPrediction = mycf(observedParameters)
priorCriteria = computeABCCriteria(priorPrediction, observedVariables)
print(priorCriteria)


# %%
# Calibrate the model with ABC
# --------------------------------------------------
# The ABC method calibrate the model by sample conditioning
observedParameterIndices = [0]
toCalibrateParameterIndices = [1, 2, 3]
observedOutputIndices = [0]
doeSize = 15000  # Size of the prior MonteCarlo sample
posteriorSampleTargetedSize = 100  # Targegeted size of the posterior conditional sample
minCvRMSE = 0.0
minNMBE = -0.005
maxCvRMSE = 0.025
maxNMBE = 0.005
n_cpus = 10
criteriaSelection = ot.Interval(
    [0, 0, minCvRMSE, minNMBE],
    [0, 0, maxCvRMSE, maxNMBE],
    [False, False, True, True],
    [False, False, True, True],
)
algo = otABCC.ABCalibration(
    cm.model,
    computeABCCriteria,
    observedParameterIndices,
    toCalibrateParameterIndices,
    observedOutputIndices,
    observedParameters,
    observedVariables,
    distributionInputs,
    doeSize,
    posteriorSampleTargetedSize,
    criteriaSelection,
    n_cpus,
)
algo.setABCCriteriaDescription(
    [r"$RMSE_{\sigma}$", r"$MBE_{\sigma}$", r"$CvRMSE_{\sigma}$", r"$NMBE_{\sigma}$"]
)
algo.run()

# %%
# Investigate the results
result = algo.getResult()
print(algo.getPriorDOE())

# %%
# draw posterior input distribution to analyse calibration
# it can be seen that :math:`\gamma` cannot be idenfied accurately but that some correlation with 
# the two other parameters are present. 
grid = result.conditionalSample.drawPosteriorInputDistribution()
fig = otv.View(grid)
fig.show()

# %%
# on the new picture, the residuals distribution of the computed optimal point (the point that maximise the posterior input distribution infered from the empiric posterior sample) is analysed. 
# the figure suggets that the discrepencies between model prediction and observed output are mostly due to measurment erros as the residuals are gaussian and centered.
print(result.getParameterMAP())
grid = result.drawResiduals()
fig = otv.View(grid)
fig.show()
grid = result.drawObservationsVsPredictions()
fig = otv.View(grid) 
fig.show()
