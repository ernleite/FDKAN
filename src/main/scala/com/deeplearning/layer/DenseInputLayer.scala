package com.deeplearning.layer

import breeze.linalg.{DenseMatrix, DenseVector, linspace, normalize}
import breeze.plot.{Figure, plot, scatter}
import com.deeplearning.Network.{generateRandomBiasFloat, generateRandomFloat}
import com.deeplearning.bspline.{BSpline, BSplineFigure, KnotPoints}
import com.deeplearning.bspline.KANControlPoints.initializeControlPoints2D
import com.deeplearning.{ActivationManager, ComputeActivation, CostManager, Network}
import com.deeplearning.samples.{CifarData, MnistData, TrainingDataSet}

import java.time.Instant

class DenseInputLayer extends InputLayer {
  var parameterSended = false
  var nablas_w_tmp = Array.empty[Float]
  var knots : DenseVector[Double] = _
  var k = 3 //degree of the spline
  var c = k + 2
  var controlPoints :Array[Array[DenseMatrix[Double]]]= _
  var ts :Array[Array[Float]]= _
  var ws = Array[Float]()
  var wb = Array[Float]()

  private var dataSet: TrainingDataSet = if (Network.trainingSample == "Mnist") {
    println("Loading Dataset MINST")
    dataSet = new MnistData()
    dataSet
  } else {
    dataSet = new CifarData()
    dataSet
  }
  def computeInputWeights(epoch:Int, correlationId: String, yLabel:Int, startIndex:Int, endIndex:Int, index:Int, layer:Int, internalSubLayer:Int,params: scala.collection.mutable.HashMap[String,String]): Array[Array[Float]] = {
    if (lastEpoch != epoch) {
      counterTraining = 0
      counterBackPropagation = 0
      counterFeedForward = 0
      lastEpoch = epoch
    }
    epochCounter = epoch
    counterTraining += 1
    val startedAt = Instant.now
    val nextLayer = layer+1

    val receiverUCs = Network.getHiddenLayers(1, "hidden")

    if (!wInitialized) {
      val arraySize = Network.InputLayerDim
      nabla_w = Array.ofDim(arraySize)
      dataSet.loadTrainDataset(Network.InputLoadMode)

      wInitialized = true
      wb =  generateRandomBiasFloat(receiverUCs)
      ws = Array.fill[Float](receiverUCs)(1.0f)

      // randomly initializing the control points
      controlPoints = Array.fill(endIndex-startIndex, receiverUCs) {
        initializeControlPoints2D(c)
      }// Convert each Seq to Array

      knots = KnotPoints.initializeKnotPoints2D(c,k)
    }

    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      val input = dataSet.getTrainingInput(index)
      val x = input.slice(startIndex + 2, endIndex + 2) //normalisation
      val v = CostManager.divide(x, 255)
      this.X += (correlationId -> v)
    }

    ts = controlPoints.indices.map (
      i => controlPoints(i).indices.map {
        j =>
          val x = this.X(correlationId)(i)
          wb(j) * ActivationManager.ComputeZ(Network.getActivationLayersType(layer), x) + ws(j) * BSpline.compute(controlPoints(i)(j),k,x,knots).toFloat
      }.toArray
    ).toArray

    //sum the same index
    val ts2 = ts.transpose.map(_.sum)
    val data = DenseVector(ts2)
    val normalizedData = normalize(data).toArray

    BSplineFigure.draw(controlPoints(0)(0),knots,k,epoch, layer,internalSubLayer)

    //split the array to be sent to layer +1 UCs
    //val splittedLayerDim = Network.HiddenLayersDim(layer)
    //val w2 = w1.grouped(w1.size/splittedLayerDim).toArray

    if (!parameterSended) {
    //  parameters("min") = weights.min.toString
    //  parameters("max") = weights.max.toString
      parameterSended = true
    }

    for (i: Int <- 0 until Network.getHiddenLayersDim(nextLayer, "hidden")) {
      val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_" + i)
      actorHiddenLayer ! ComputeActivation.ComputeZ(epoch, correlationId, yLabel, Network.MiniBatchRange, normalizedData, i, nextLayer, Network.InputLayerDim, params, Array.empty[Float])
    }
    null
  }

  def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Float, internalSubLayer: Int, fromInternalSubLayer: Int, params: scala.collection.mutable.HashMap[String,String]): Boolean = {
    counterBackPropagation += 1
    minibatch(correlationId) += 1
    //params("eventBP") =  (params("eventBP").toInt + 1).toString

    if (!backPropagateReceived.contains(correlationId)) {
      val fromArraySize = Network.getHiddenLayersDim(1, "hidden")
      backPropagateReceived += (correlationId -> true)
      //nablas_w(correlationId) = Array.ofDim(fromArraySize)
    }
    var startedAt = Instant.now
    if (nablas_w_tmp.isEmpty)
      nablas_w_tmp =  CostManager.dotProduct3(delta.length, delta, X(correlationId)).flatten
    else
      nablas_w_tmp = CostManager.sum2(nablas_w_tmp, CostManager.dotProduct3(delta.length, delta, X(correlationId)).flatten)
    //nablas_w(correlationId) =CostManager.dotProduct3(delta.length, delta, X(correlationId)).flatten
    //////////////////////////nablas_w(correlationId)(fromInternalSubLayer) = dot
    // how many calls would we received
    val callerSize = Network.getHiddenLayersDim(1, "hidden")

    // check if we reach the last mini-bacth
    //context.log.info("Receiving from bakpropagation")
    if ((Network.MiniBatch * callerSize) == minibatch.values.sum) {
      val a = Network.rangeInitStart
      val b = Network.rangeInitEnd
      parameterSended = false
      params("min") = "0"
      params("max") = "0"

      /*
      val flatten = nablas_w.values.toArray
      val reduce = flatten.transpose.map(_.sum)
      val nablaflaten = reduce

       */
      val tmp2 = CostManager.matMulScalar(learningRate / Network.MiniBatch,nablas_w_tmp)
      val tmp1 = CostManager.matMulScalar(1 - learningRate * (regularisation / nInputs), this.ts.flatten)

      val ttt = ts.map(x=> {
        x.map(y=> {
          val findKnot = BSpline.findSpan(y.toDouble,k,c,knots.toArray)
          val test2 = 1
        })
      })

      val updated = CostManager.minus2(tmp1, tmp2)
     // weights =  updated
//      weights = CostManager.minus2(tmp1, tmp2)

      //val scalling = CostManager.scalling(weights, Network.InputLayerDim, parameters("weighted_min").toFloat, parameters("weighted_min").toFloat)
      //weights = scalling

      //println("---------------------------------------------------------------")
      //println("Back-propagation event IL (0): Weights updated")

      backPropagateReceived.clear()
      minibatch.clear()
      weighted.clear()
      nablas_w.clear()
      counterBackPropagation=0
      nablas_w_tmp = Array.empty[Float]

      this.X.clear()
      true
    }
    else
      false
  }

  def FeedForwardTest(correlationId: String, startIndex: Int, endIndex: Int, index: Int, internalSubLayer: Int, layer: Int):  Array[Array[Float]] = {
    if (!wTest) {
      wTest = true
      //Temporary
      // read shard data from data lake
      dataSet.loadTestDataset(Network.InputLoadMode)
    }

    val nextLayer = layer + 1
    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      //for (i <- index until (index+1)) {
      val input = dataSet.getTestInput(index)
      val x = input.slice(startIndex + 2, endIndex + 2) //normalisation
      val v = CostManager.divide(x, 255)
      this.XTest += (correlationId -> v)
      wTest = true
    }

    val x = Array.fill(Network.getHiddenLayers(1, "hidden"))(this.XTest(correlationId)).flatten
    val w1 = CostManager.dotProduct(weights, x)
    //split the array to be sent to layer +1 UCs
    val splittedLayerDim = Network.HiddenLayersDim(layer)
    val w2 = w1.grouped(w1.size / splittedLayerDim).toArray
    for (i: Int <- 0 until Network.getHiddenLayersDim(nextLayer, "hidden")) {
      val arr = w2(i)
      val l = this.XTest(correlationId).length
      val weighted = arr.grouped(l).toArray.map(_.sum)
      val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_" + i)
      actorHiddenLayer ! ComputeActivation.FeedForwardTest(correlationId, weighted, i, nextLayer, Network.InputLayerDim)
    }

    weighted -= (correlationId)
    minibatch -= (correlationId)
    XTest -= (correlationId)
    w2
  }
}
