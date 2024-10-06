"""
Reconcile data for the HVAC Splitter
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


# # Reconciliation de donnée avec OpenModelica et OpenTURNS
# L'objet est de tester la classe DataReconciliationOM fournissant un wrapper python de la reconciliation de donnée réalisée dans OpenModelica.
# Les résultats de la réconciliation de données sont accessibles au format OpenTURNS (distribution, matrices) pour être visualisés et utilisés dans des études d'incertitudes (propagation, calage par exemple).

# Import des modules nécessaires
import otDataReconciliation as otDataR
import openturns as ot
import openturns.viewer as otv
import matplotlib.pyplot as plt
from IPython.display import Latex, display
from IPython import get_ipython
import os

# Affichage graphiques en ligne si utilisation Ipython
try:
    get_ipython().run_line_magic("matplotlib", "inline")
except:
    pass

# Le modèle jouet est ici l'exemple d'un gaine de ventilation se séparant
# $$ Q_3 = Q_1 + Q_2 $$
# $$ Q_3 c_p T_3 = Q_1 c_p T_1 + Q_2 c_p T_2 $$
# Les débits sont ici en $m^3.h^{-1}$ et les températures en $degC$
modelName = "TestDataR_HVACSplitter"

# Construction de la loi à priori des mesures à réconciliées
# Une corrélation de 0.5 est spécifiée entre $Q_1$ et $Q_2$
# Dans le cadre de la réconciliation de donnée mis en oeuvre dans OpenModelica, la loi à priori doit être une loi normale multivariée.
VarNames = ot.Description(["Q1", "Q2", "Q3", "T1", "T2", "T3"])
priorMeasurements = ot.Point([130, 210, 360, 20, 30, 22])
priorMeasurementsStandardDeviation = ot.Point([10, 20, 20, 0.3, 0.3, 0.3])
correlationMatrix = ot.CorrelationMatrix(
    len(VarNames)
)  # Initialisé à la matrice identité
correlationMatrix[0, 1] = 0.5  # Spécification du coefficient de corrélation souhaité
priorDistribution = ot.Normal(
    priorMeasurements, priorMeasurementsStandardDeviation, correlationMatrix
)

# La matrice étant symmétrique, le coefficient [1,0] est automatiquement mis à jour
display(correlationMatrix)

# Construction de la classe réconciliation de données
casePath = "./"
simuMatPath = "SIMU_MAT_0"
covarianceMatrix = priorDistribution.getCovariance()
dataR = otDataR.DataReconciliationOM(
    modelName, VarNames, priorMeasurements, covarianceMatrix, casePath, simuMatPath
)

# Visualisation de la distribution multivariée des mesures avant réconciliation
priorDistribution = dataR.getPriorDistribution()
grid = priorDistribution.drawDistributionGridPDF()
fig = otv.View(grid)

# Lancer la réconciliation de données
dataR.reconcileData()

# # Analyse des résultats
df = dataR.getLocalResultsAsDataFrame()
display(df)

# Visualisation de la distribution multivariée des mesures réconciliées, marginales 1D et 2D (visualisation de la dépendance à posteriori)
posteriorDistribution = dataR.getPosteriorDistribution(index=[0, 1, 2])
grid = posteriorDistribution.drawDistributionGridPDF()
fig = otv.View(grid)

# Il est également possible de récupérer la matrice de covariance à posteriori
posteriorCovM = dataR.getCovarianceMatrixAsDataFrame()
display(posteriorCovM)

# Accès à la matrice de corrélation reconciliée des trois débits
posteriorCorrelationMatrix = dataR.getPosteriorCorrelationMatrix(index=[0, 1, 2])
posteriorCorrelationMatrixDF = dataR.getPosteriorCorrelationMatrixAsDataFrame(
    index=[0, 1, 2]
)
display(posteriorCorrelationMatrixDF)

# Visualisation de l'effet de la réconciliation de données sur les marginales
grid = dataR.drawPriorPosteriorMarginalsDistribution()
fig = otv.View(grid)
