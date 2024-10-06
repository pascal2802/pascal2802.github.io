"""
Reconcile data for the HVAC Splitter
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


# # Reconciliation de donnée avec OpenModelica et OpenTURNS
# L'objet est de tester la classe DataReconciliationOM fournissant un wrapper python de la reconciliation de donnée réalisée dans OpenModelica.
# Les résultats de la réconciliation de données sont accessibles au format OpenTURNS (distribution, matrices) pour être visualisées et utilisées dans des études d'incertitudes (propagation, calage par exemple).

# Import des modules nécessaires

import openturns as ot
import openturns.viewer as otv
from IPython import get_ipython
from IPython.display import display

import otDataReconciliation as otDataR

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

# Construction de la classe réconciliation de donnée
VarNames = ot.Description(["Q1", "Q2", "Q3", "T1", "T2", "T3"])
priorMeasurements = ot.Point([130, 210, 360, 20, 30, 22])
covM = ot.CovarianceMatrix([
    [100, 100, 0, 0, 0, 0],
    [100, 400, 0, 0, 0, 0],
    [0, 0, 400, 0, 0, 0],
    [0, 0, 0, 0.3**2, 0, 0],
    [0, 0, 0, 0, 0.3**2, 0],
    [0, 0, 0, 0, 0, 0.3**2],
])
casePath = "./"
simuMatPath = "SIMU_MAT_0"
dataR = otDataR.DataReconciliationOM(modelName, VarNames, priorMeasurements, covM, casePath, simuMatPath)

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
posteriorDistribution = dataR.getPosteriorDistribution()
grid = posteriorDistribution.drawDistributionGridPDF()
fig = otv.View(grid)

# Il est également possible de récupérer la matrice de covariance à posteriori
posteriorCovM = dataR.getCovarianceMatrixAsDataFrame()
display(posteriorCovM)

# Simulation de l'effet de la réconciliation de donnée sur les marginales
grid = dataR.drawPriorPosteriorMarginalsDistribution()
fig = otv.View(grid)
