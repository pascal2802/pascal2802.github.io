
.. DO NOT EDIT.
.. THIS FILE WAS AUTOMATICALLY GENERATED BY SPHINX-GALLERY.
.. TO MAKE CHANGES, EDIT THE SOURCE PYTHON FILE:
.. "auto_examples/plot_calibration_chaboche_model.py"
.. LINE NUMBERS ARE GIVEN BELOW.

.. only:: html

    .. note::
        :class: sphx-glr-download-link-note

        :ref:`Go to the end <sphx_glr_download_auto_examples_plot_calibration_chaboche_model.py>`
        to download the full example code.

.. rst-class:: sphx-glr-example-title

.. _sphx_glr_auto_examples_plot_calibration_chaboche_model.py:


Calibrate Chaboche model using ABCCalibration Class
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. GENERATED FROM PYTHON SOURCE LINES 7-18

This example aims to illustrate the use of the ABCCalibration class with the Chaboche model
We recall that the Chaboche model is the following :

.. math::
  \sigma = G(\epsilon,R,C,\gamma) = R + \frac{C}{\gamma} (1-\exp(-\gamma\epsilon))

where:

- :math:`\epsilon` is the strain,
- :math:`\sigma` is the stress (Pa),
- :math:`R`, :math:`C`, :math:`\gamma` are the parameters.

.. GENERATED FROM PYTHON SOURCE LINES 21-22

| Loading python modules

.. GENERATED FROM PYTHON SOURCE LINES 24-38

.. code-block:: Python

    import openturns as ot
    from openturns.usecases import chaboche_model
    import math
    import imp
    import otABCCalibration.ABC_ClassProto as otABCC
    import openturns.viewer as otv
    import matplotlib.pyplot as plt
    import openturns.viewer as otv


    imp.reload(otABCC)

    # ot.Log.Show(ot.Log.NONE)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    /home/d54380/Logiciels/OpenTURNS/otABCCalibration/doc/examples/plot_calibration_chaboche_model.py:27: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp

    <module 'otABCCalibration.ABC_ClassProto' from '/home/d54380/Logiciels/OpenTURNS/otABCCalibration/otABCCalibration/ABC_ClassProto.py'>



.. GENERATED FROM PYTHON SOURCE LINES 39-41

Define the Observations
==================================================

.. GENERATED FROM PYTHON SOURCE LINES 41-47

.. code-block:: Python

    cm = chaboche_model.ChabocheModel()
    print(cm.data)
    observedParameters = cm.data[:, 0]  # Strain
    observedVariables = cm.data[:, 1]  # Stress
    numberOfObservations = cm.data.getSize()





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

        [ Strain      Stress (Pa) ]
    0 : [ 0           7.56e+08    ]
    1 : [ 0.0077      7.57e+08    ]
    2 : [ 0.0155      7.85e+08    ]
    3 : [ 0.0233      8.19e+08    ]
    4 : [ 0.0311      8.01e+08    ]
    5 : [ 0.0388      8.42e+08    ]
    6 : [ 0.0466      8.49e+08    ]
    7 : [ 0.0544      8.79e+08    ]
    8 : [ 0.0622      8.85e+08    ]
    9 : [ 0.07        8.96e+08    ]




.. GENERATED FROM PYTHON SOURCE LINES 48-52

Set Calibration prior distribution
--------------------------------------------------
The prior observed parameters uncertainty distribution parameter is set
random uncertainty will be add to the observed parameters sample while evaluated ABC DOE.

.. GENERATED FROM PYTHON SOURCE LINES 52-56

.. code-block:: Python

    StrainUdistribution = ot.Normal(0, 0.001)
    distributionUObsParameters = ot.ComposedDistribution([StrainUdistribution])
    distributionUObsParameters.setDescription([r"$U_{\varepsilon}$"])








.. GENERATED FROM PYTHON SOURCE LINES 57-58

Define the prior joint distribution of the parameter to calibrate :math:`\pi(\theta)`

.. GENERATED FROM PYTHON SOURCE LINES 58-70

.. code-block:: Python


    Rdistribution = ot.Uniform(700.0e6, 800.0e6)
    Cdistribution = ot.Uniform(1000.0e6, 4000.0e6)
    gammaDistribution = ot.Uniform(1.0, 10.0)
    distributionParameters = ot.ComposedDistribution(
        [Rdistribution, Cdistribution, gammaDistribution]
    )
    distributionParameters.setDescription(["R", "C", r"$\gamma$"])

    thetaPrior = distributionParameters.getMean()
    print(thetaPrior)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [7.5e+08,2.5e+09,5.5]




.. GENERATED FROM PYTHON SOURCE LINES 71-72

Build a joint distribution between parameters to calibrate prior and observed parameter uncertainty prior

.. GENERATED FROM PYTHON SOURCE LINES 72-79

.. code-block:: Python


    distributionInputs = ot.ComposedDistribution(
        [StrainUdistribution, Rdistribution, Cdistribution, gammaDistribution]
    )
    distributionInputs.setDescription([r"$U_{\varepsilon}$", "R", "C", r"$\gamma$"])









.. GENERATED FROM PYTHON SOURCE LINES 80-83

Set the calibration criteria
==================================================
modeller need to define the computation of the criteria to define a calibrated model based on the returned sample by the evaluation of all the observation point

.. GENERATED FROM PYTHON SOURCE LINES 83-122

.. code-block:: Python



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









.. GENERATED FROM PYTHON SOURCE LINES 123-124

test the function with the :math:`\theta_{prior}` computed above

.. GENERATED FROM PYTHON SOURCE LINES 124-130

.. code-block:: Python

    calibratedIndices = [1, 2, 3]
    mycf = ot.ParametricFunction(cm.model, calibratedIndices, thetaPrior)
    priorPrediction = mycf(observedParameters)
    priorCriteria = computeABCCriteria(priorPrediction, observedVariables)
    print(priorCriteria)





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

    [9.91815e+06,-227245,0.0119944,-0.000274815]




.. GENERATED FROM PYTHON SOURCE LINES 131-134

Calibrate the model with ABC
--------------------------------------------------
The ABC method calibrate the model by sample conditioning

.. GENERATED FROM PYTHON SOURCE LINES 134-167

.. code-block:: Python

    observedParameterIndices = [0]
    toCalibrateParameterIndices = [1, 2, 3]
    doeSize = 15000  # Size of the prior MonteCarlo sample
    posteriorSampleTargetedSize = 100  # Targegeted size of the posterior conditional sample
    minCvRMSE = 0.0
    minNMBE = -0.003
    maxCvRMSE = 0.015
    maxNMBE = 0.003
    criteriaSelection = ot.Interval(
        [0, 0, minCvRMSE, minNMBE],
        [0, 0, maxCvRMSE, maxNMBE],
        [False, False, True, True],
        [False, False, True, True],
    )
    algo = otABCC.ABCCalibration(
        cm.model,
        computeABCCriteria,
        observedParameterIndices,
        toCalibrateParameterIndices,
        observedParameters,
        observedVariables,
        distributionUObsParameters,
        distributionParameters,
        distributionInputs,
        doeSize,
        posteriorSampleTargetedSize,
        criteriaSelection,
    )
    algo.setABCCriteriaDescription(
        [r"$RMSE_{\sigma}$", r"$MBE_{\sigma}$", r"$CvRMSE_{\sigma}$", r"$NMBE_{\sigma}$"]
    )
    algo.run()








.. GENERATED FROM PYTHON SOURCE LINES 168-169

Investigate the results

.. GENERATED FROM PYTHON SOURCE LINES 169-171

.. code-block:: Python

    print(algo.getPriorDOE())





.. rst-class:: sphx-glr-script-out

 .. code-block:: none

            [ $U_{\varepsilon}$ R                 C                 $\gamma$          ]
        0 : [  0.000608202       7.29261e+08       3.63531e+09       3.21978          ]
        1 : [ -0.00126617        7.98908e+08       1.25791e+09       7.51201          ]
        2 : [ -0.000438266       7.53989e+08       2.83859e+09       9.51469          ]
    ...
    14997 : [ -8.48901e-05       7.98416e+08       1.15237e+09       3.98828          ]
    14998 : [  0.000363987       7.84281e+08       2.36053e+09       6.10189          ]
    14999 : [  0.000211898       7.21996e+08       2.33355e+09       1.37144          ]




.. GENERATED FROM PYTHON SOURCE LINES 172-173

draw posterior input distribution to analyse calibration

.. GENERATED FROM PYTHON SOURCE LINES 173-176

.. code-block:: Python

    grid = algo.result.conditionalSample.drawPosteriorInputDistribution()
    fig = otv.View(grid)
    fig.show()



.. image-sg:: /auto_examples/images/sphx_glr_plot_calibration_chaboche_model_001.png
   :alt: Conditional Sample : 211 out of 15000   0.000 < $CvRMSE_{\sigma}$ < 0.015   -0.003 < $NMBE_{\sigma}$ < 0.003 , Spearman : -0.26, Spearman : 0.00, Spearman : -0.01, Spearman : -0.26, Spearman : -0.84, Spearman : -0.25, Spearman : 0.00, Spearman : -0.84, Spearman : 0.68, Spearman : -0.01, Spearman : -0.25, Spearman : 0.68
   :srcset: /auto_examples/images/sphx_glr_plot_calibration_chaboche_model_001.png
   :class: sphx-glr-single-img






.. rst-class:: sphx-glr-timing

   **Total running time of the script:** (0 minutes 6.663 seconds)


.. _sphx_glr_download_auto_examples_plot_calibration_chaboche_model.py:

.. only:: html

  .. container:: sphx-glr-footer sphx-glr-footer-example

    .. container:: sphx-glr-download sphx-glr-download-jupyter

      :download:`Download Jupyter notebook: plot_calibration_chaboche_model.ipynb <plot_calibration_chaboche_model.ipynb>`

    .. container:: sphx-glr-download sphx-glr-download-python

      :download:`Download Python source code: plot_calibration_chaboche_model.py <plot_calibration_chaboche_model.py>`