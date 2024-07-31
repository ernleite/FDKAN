package com.deeplearning.layer

import breeze.linalg.{DenseMatrix, DenseVector, normalize}
import com.deeplearning.CostManager._
import com.deeplearning.Network.generateRandomBiasFloat
import com.deeplearning.bspline.KANControlPoints.initializeControlPoints2D
import com.deeplearning.bspline.{BSpline, BSplineFigure, KnotPoints}
import com.deeplearning.{ActivationManager, ComputeActivation, ComputeOutput, CostManager, LayerManager, Network}
import com.deeplearning.bspline.ControlPointUpdater

import java.time.{Duration, Instant}


class DenseWeightedLayer extends WeightedLayer {
  var activationsLength:Int = 0
  var parameterSended = false
  val parameters = scala.collection.mutable.HashMap.empty[String, String]
  var deltas =  Array[Float]()
  var deltas2 = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  var shardReceived = scala.collection.mutable.HashMap.empty[String, Int]
  var inProgress = scala.collection.mutable.HashMap.empty[String, Boolean]
  var nablas_w_tmp = Array.empty[Float]
  var knots : DenseVector[Double] = _
  var k = 3 //degree of the spline
  var c = k + 1
  var controlPoints :Array[Array[DenseMatrix[Double]]]= _
  var ts = scala.collection.mutable.HashMap.empty[String,Array[Array[Float]]]
  var tsTest : Array[Array[Float]] = null

  var ws = Array[Float]()
  var wb = Array[Float]()

  private var ucIndex = 0
  def Weights(epoch: Int, correlationId: String, yLabel: Int, trainingCount: Int, activations: Array[Float], shards: Int, internalSubLayer: Int, layer: Int, params : scala.collection.mutable.HashMap[String,String]) : Array[Array[Float]] = {
    if (this.lastEpoch != epoch) {
      this.counterTraining = 0
      this.counterBackPropagation = 0
      this.counterFeedForward = 0
      this.lastEpoch = epoch
    }

    val startedAt = Instant.now
    val nextLayer = layer + 1
    this.counterTraining += 1

    var hiddenLayerStep = 0
    var arraySize = 0
    var receiverUCs = 0

    if (!LayerManager.IsLast(nextLayer)) {
      hiddenLayerStep = LayerManager.GetDenseActivationLayerStep(nextLayer)
      arraySize = Network.getHiddenLayersDim(nextLayer, "hidden")
      receiverUCs = Network.getHiddenLayers(nextLayer, "hidden")
    }
    else {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      arraySize = Network.OutputLayerDim
      receiverUCs = Network.OutputLayer
    }

    var split = 0
    if (!LayerManager.IsLast(nextLayer))
      split = Network.getHiddenLayers(nextLayer, "hidden")
    else
      split = Network.OutputLayer

    if (!this.wInitialized) {
      ucIndex = internalSubLayer
      this.activationsLength = activations.length
      this.weights = Array.ofDim(arraySize)
      this.nabla_w = Array.ofDim(arraySize)
      //this.weights = generateRandomFloat(split*activationsLength)
      this.wInitialized = true
      parameters += ("min" -> "0")
      parameters += ("max" -> "0")
      parameters += ("weighted_min" -> "0")
      parameters += ("weighted_max" -> "0")
      wb =  generateRandomBiasFloat(receiverUCs)
      ws = Array.fill[Float](receiverUCs)(1.0f)

      // randomly initializing the control points
      controlPoints = Array.fill(activationsLength, receiverUCs) {
        initializeControlPoints2D(c)
      }
      knots = KnotPoints.initializeKnotPoints2D(c,k)
    }

    //this.activation += (correlationId -> activations)
    if (!this.minibatch.contains(correlationId)) {
      this.minibatch += (correlationId -> 0)
      this.messagePropagateReceived += (correlationId -> 0)
      this.fromInternalReceived += (correlationId -> 0)
      this.activation += (correlationId -> activations)
      this.deltas = Array.fill(activationsLength)(0f)
      this.ts += (correlationId -> null)
    }

    if (!inProgress.contains(correlationId)) {
      inProgress += (correlationId -> true)
      shardReceived += (correlationId -> 0)
    }
    shardReceived(correlationId) +=1

    if (shardReceived(correlationId) < shards && inProgress(correlationId)) {
      activation(correlationId) = CostManager.sum2(activation(correlationId), activations)
      null
    }
    else {
      val callersize = Network.getHiddenLayersDim(layer, "weighted")
      activation(correlationId) = CostManager.sum2(activation(correlationId), activations)
      inProgress(correlationId) = false
      activation(correlationId) = normalize(DenseVector(activation(correlationId))).toArray
      ts(correlationId) = controlPoints.indices.map (
        i => controlPoints(i).indices.map {
          j =>
            val x = activation(correlationId)(i)
            val y = ActivationManager.ComputeZ(Network.getActivationLayersType(layer), x) + BSpline.compute(controlPoints(i)(j),k,x,knots).toFloat
            /*
            if (y>1.0f) {
              println("-----------------------------------------")
              println("Threshold " + y )
              println("-----------------------------------------")
            }

             */
            y
        }.toArray
      ).toArray

      //sum the same index
      val ts2 =   ts(correlationId).transpose.map(_.sum)
      val data = DenseVector(ts2)
      val normalizedData = normalize(data).toArray

     // BSplineFigure.draw(controlPoints(0)(0),knots,k,epoch, layer,internalSubLayer)

      val endedAt = Instant.now
      val duration = Duration.between(startedAt, endedAt).toMillis

      if (counterTraining % Network.minibatchBuffer == 0 && Network.debug) {
        println("-------------------------------------------")
        println("Weighted feedforward duration : " + duration)
      }

      if (!parameterSended) {
        //    parameters("min") = weights.min.toString
        //    parameters("max") = weights.max.toString
        parameterSended = true
      }

      if (!LayerManager.IsLast(nextLayer)) {
        for (i: Int <- 0 until Network.getHiddenLayersDim(nextLayer, "hidden")) {
          val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_" + i)
          actorHiddenLayer ! ComputeActivation.ComputeZ(epoch, correlationId, yLabel, trainingCount, normalizedData, i, nextLayer, callersize,params, weights)
        }
      }
      else if (LayerManager.IsLast(nextLayer)) {
        for (i <- 0 until Network.OutputLayerDim) {
          val actorOutputLayer = Network.LayersOutputRef("outputLayer_" + i)
          actorOutputLayer ! ComputeOutput.Compute(epoch, correlationId, yLabel, trainingCount, normalizedData, i, layer+1, callersize, params)
        }
      }
      null
    }
  }

  def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Int, layer: Int, internalSubLayer: Int, fromInternalSubLayer: Int, params : scala.collection.mutable.HashMap[String,String], applyGrads:Boolean) : Boolean = {
    var hiddenLayerStep = 0
    var arraySize = 0

    val fromLayer = layer + 1
    var fromArraySize = 0
    val nextlayer = layer + 1

    if ((fromLayer - 1) == Network.HiddenLayersDim.length * 2) {
      fromArraySize = Network.OutputLayerDim
    }
    else {
      fromArraySize = Network.getHiddenLayersDim(nextlayer, "hidden")
    }

    if (applyGrads) {
      applyGradients(arraySize, learningRate, fromInternalSubLayer, nInputs, regularisation)
      return true
    }
    else {
      this.minibatch(correlationId) += 1
      this.messagePropagateReceived(correlationId) += 1
      this.counterBackPropagation += 1
      //println("Message " + " " + messagePropagateReceived(correlationId))

      if (!backPropagateReceived.contains(correlationId)) {
        backPropagateReceived += (correlationId -> true)
      }

      var callerSize = 0
      var group = 0
      if (LayerManager.IsLast(nextlayer)) {
        hiddenLayerStep = LayerManager.GetOutputLayerStep()
        arraySize = Network.OutputLayerDim
        group = LayerManager.GetDenseWeightedLayerStep(layer)
      }
      else if (LayerManager.IsFirst(layer - 1)) {
        arraySize = Network.InputLayerDim
        group = Network.InputLayerDim
      }
      else {
        arraySize = Network.getHiddenLayersDim(layer, "weighted")
        group = LayerManager.GetDenseWeightedLayerStep(layer)
      }

      val newdelta = delta

      if (nablas_w_tmp.isEmpty) {
        val tt = controlPoints.indices.map (
          i => controlPoints(i).indices.map {
            val x = activation(correlationId)
            val siluPrime = ActivationManager.ComputePrime(Network.getActivationLayersType(layer), x)
            j =>
              val tttt =   ts(correlationId)(i)(j)
              val splinePrime = BSpline.bsplineDerivative(  ts(correlationId)(i)(j), controlPoints(i)(j),knots,k)
              val finald = siluPrime(j) + splinePrime.toFloat
              finald
          }.toArray
        ).toArray

        nablas_w_tmp = CostManager.matMul(tt.transpose, delta)
      }
      else {
        val tt = controlPoints.indices.map (
          i => controlPoints(i).indices.map {
            val x = activation(correlationId)
            val siluPrime = ActivationManager.ComputePrime(Network.getActivationLayersType(layer), x)
            j =>
              val splinePrime = BSpline.bsplineDerivative(  ts(correlationId)(i)(j), controlPoints(i)(j),knots,k)
              val finald = siluPrime(j) +splinePrime.toFloat
              finald
          }.toArray
        ).toArray
        nablas_w_tmp = CostManager.sum2(nablas_w_tmp,CostManager.matMul(tt.transpose, delta) )
      }

      var delta2 = Array.empty[Float]
      if (arraySize == 1) {
        if (Network.debugActivity)
          println("WL UNIQUE UPDATED  Delta Layer:" + layer + " Index: " + ucIndex + " FROM: " + fromInternalSubLayer)
        val compute = CostManager.dotProduct4(  ts(correlationId).transpose, delta)
        delta2 =compute
      }
      else {
        if (Network.debugActivity)
          println("WL UPDATED Delta Layer:" + layer + " Index: " + ucIndex + " FROM: " + fromInternalSubLayer)
        //val weightsDeltas = weights.grouped(this.weights.size / arraySize).toArray
        val weightsDelta = this.activation(correlationId)
        val split = weightsDelta.grouped(this.weights.size/arraySize).toArray
        val toUpdate = split(fromInternalSubLayer).grouped(group).toArray
        val upd = CostManager.dotProduct4(toUpdate, delta)
        delta2 =  upd
      }

      //callerSize = arraySize * Network.getHiddenLayersDim(nextlayer, "weighted")
      // only send when all the call from the layer +1 are received
      if (fromArraySize == messagePropagateReceived(correlationId)) {
        fromInternalReceived(correlationId) += 1

        if (layer > 1 && Network.getHiddenLayersType(layer - 1, "hidden") == "Conv2d") {
          val hiddenLayerRef = Network.LayersHiddenRef("hiddenLayer_" + (layer - 1) + "_" + ucIndex)
          hiddenLayerRef ! ComputeActivation.BackPropagate(correlationId, newdelta, learningRate, regularisation, nInputs, layer - 1, ucIndex, params)
        }
        else {
          val hiddenLayerRef = Network.LayersHiddenRef("hiddenLayer_" + (layer - 1) + "_" + ucIndex)
          if (Network.debugActivity)
            println("Send AL " + (layer - 1) + " " + ucIndex)
          //val partialDelta = deltas.grouped(arraySize).toArray
          hiddenLayerRef ! ComputeActivation.BackPropagate(correlationId, delta2, learningRate, regularisation, nInputs, layer - 1, ucIndex, params)
          /*
          var arraySize2 = Network.getHiddenLayersDim(layer)
          for (i <- 0 until arraySize2) {
            val hiddenLayerRef = Network.LayersHiddenRef("hiddenLayer_" + (layer - 1) + "_" + i)
            if (Network.debugActivity)
              println("Send AL " + (layer - 1) + " " + i)

            if (arraySize == 1) {
              hiddenLayerRef ! ComputeActivation.BackPropagate(correlationId, deltas, learningRate, regularisation, nInputs, layer - 1, internalSubLayer, scala.collection.mutable.HashMap.empty[String, String])
            }
            else {
              val split = deltas.grouped(delta.size).toArray
              hiddenLayerRef ! ComputeActivation.BackPropagate(correlationId, split(i), learningRate, regularisation, nInputs, layer - 1, internalSubLayer, scala.collection.mutable.HashMap.empty[String, String])
            }
          }

           */
        }
      }

      //println("Backpropagation WL ("+layer+")")

      // check if we reach the last mini-bacth

      val verticalUCs = Network.getHiddenLayers(layer, "weigthed")

      if (Network.MiniBatch *fromArraySize  == minibatch.values.sum) {
        parameterSended = false
        parameters("min") = "0"
        parameters("max") = "0"
        //val mav = Normalisation.getMeanAndVariance(activation(correlationId))
        //activation(correlationId) = Normalisation.batchNormalize(activation(correlationId), mav._1, mav._3, 0.1f, 0.1f)
        // println("Apply gradients layer " + layer + " " + internalSubLayer + " " + fromInternalSubLayer)
        //synchronized()

        /*

        //val scalling = CostManager.scalling(weights, Network.getHiddenLayersDim(layer), wei("weighted_min").toFloat, params("max").toFloat)
        //weights = scalling

        if (counterTraining % Network.minibatchBuffer == 0 && Network.debug) {
          for (j <- weights.indices) {
            if (Network.debugActivity)
              println("W Layer (" + layer + "): " + weights(j) )
          }
          println("Params : " + parameters("weighted_min") + " " + parameters("weighted_max"))
          println("--------------------------------------------------------------------")
        }

        //println("---------------------------------------------------------------")
        //println("Back-propagation event WL ("+layer+"): Weights updated")
        */
        //SynchronizeWeights(layer, internalSubLayer)
        //applyGradients(callerSize, learningRate, fromInternalSubLayer, nInputs, regularisation)
        /*
        val flatten = nablas_w.values.toArray
        val reduce = flatten.transpose.map(_.sum)
        val nablaflaten = reduce
        */
        val tmp1 = CostManager.matMulScalar(1 - learningRate * (regularisation / nInputs), this.ts(correlationId).flatten)
        val tmp2 = CostManager.matMulScalar(learningRate / Network.MiniBatch,nablas_w_tmp)
        val tmp3 = tmp1.grouped(group).toArray
        val tmp4 = CostManager.minus2(tmp3, tmp2)
        val tmp5 = CostManager.divide(tmp4, tmp4.length)
        //println("Before : " +  controlPoints(0)(0).toArray(4) + " "  + controlPoints(0)(0).toArray(5)+ " "  + controlPoints(0)(0).toArray(6)+ " "  + controlPoints(0)(0).toArray(7))
        val ttt =   ts(correlationId).zipWithIndex.map {
          case (element, i) =>
            element.zipWithIndex.map {
              case (subelement, j) => {
               // println("Before : " + controlPoints(i)(j))
                controlPoints(i)(j) = ControlPointUpdater.updateControlPoints(  ts(correlationId)(i)(j), knots, controlPoints(i)(j), k, tmp5(i+j)).toDenseMatrix
               // println("After : " + controlPoints(i)(j))
              }
            }
        }
        //println("After : " + controlPoints(0)(0).toArray(4) + " "  + controlPoints(0)(0).toArray(5)+ " "  + controlPoints(0)(0).toArray(6)+ " "  + controlPoints(0)(0).toArray(7))
        this.ts -= (correlationId)
        fromInternalReceived.clear()
        activation.clear()
        backPropagateReceived.clear()
        messagePropagateReceived.clear()
        minibatch.clear()
        nablas_w.clear()
        weighted.clear()
        deltas2.clear()
        nablas_w_tmp = Array.empty[Float]
        this.deltas = Array.fill(activationsLength)(0f)
        true
      }
    else
      false
    }
  }

  def applyGradients(splitted:Int, learningRate:Float, fromInternalSubLayer:Int, nInputs:Int, regularisation:Float) : Unit = {
    val flatten = nablas_w.values.toArray
    val reduce = flatten.transpose.map(_.sum)
    val nablaflaten = reduce
    val n2 = nablaflaten.grouped(weights.size / splitted).toArray

    val tmp2 = CostManager.matMulScalar(learningRate / Network.MiniBatch, n2(fromInternalSubLayer))
    val w2 = this.weights.grouped(weights.size / splitted).toArray

    for (z <- 0 until w2.indices.length) {
      val tmp = w2(z)
      val tmp1 = CostManager.matMulScalar(1 - learningRate * (regularisation / nInputs), tmp)
      val updated = CostManager.minus2(tmp1, tmp2)
      w2.update(z, updated)
    }
    weights =  w2.flatten
  }

  override def FeedForwardTest(correlationId: String, activations: Array[Float], ucIndex: Int, layer: Int): Array[Array[Float]] =  {
    val nextLayer = layer + 1
    counterFeedForward += 1

    var hiddenLayerStep = 0
    var arraySize = 0

    if (!LayerManager.IsLast(nextLayer)) {
      hiddenLayerStep = LayerManager.GetDenseActivationLayerStep(nextLayer)
      arraySize = Network.getHiddenLayersDim(nextLayer, "hidden")
    }
    else {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      arraySize = Network.OutputLayerDim
    }
    var split = 0
    if (!LayerManager.IsLast(nextLayer))
      split = Network.getHiddenLayers(nextLayer, "hidden")
    else
      split = Network.OutputLayer

    tsTest = controlPoints.indices.map (
      i => controlPoints(i).indices.map {
        j =>
          val x = activations(i)
          ActivationManager.ComputeZ(Network.getActivationLayersType(layer), x)  * BSpline.compute(controlPoints(i)(j),k,x,knots).toFloat
      }.toArray
    ).toArray

    //sum the same index
    val ts2 =   tsTest.transpose.map(_.sum)
    val data = DenseVector(ts2)
    val normalizedData = normalize(data).toArray
    val callersize = Network.getHiddenLayersDim(layer, "weighted")

    if (!LayerManager.IsLast(nextLayer)) {
      for (i: Int <- 0 until arraySize) {
        val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + nextLayer + "_" + i)
        actorHiddenLayer ! ComputeActivation.FeedForwardTest( correlationId,  normalizedData, i, nextLayer, callersize)
      }
    }
    else if (LayerManager.IsLast(nextLayer)) {
      for (i <- 0 until arraySize) {
        val actorOutputLayer = Network.LayersOutputRef("outputLayer_" + i)
        actorOutputLayer ! ComputeOutput.FeedForwardTest(correlationId, normalizedData, i, nextLayer, callersize)
      }
    }

    activation -= correlationId
    weighted -= correlationId
    minibatch -= correlationId

    null
  }
  override def getNeighbor(localIndex:Int): Array[Float] = {
    null
  }
  override def SynchronizeWeights(layer:Int, subLayer:Int): Unit = {
    return
    for (i <- 0 until Network.getHiddenLayersDim(layer, "weighted") if i != subLayer) {
      val actorHiddenLayer = Network.LayersHiddenRef("hiddenLayer_" + layer + "_" + i)
      //val sync = actorHiddenLayer ! getNeighbor(i)
    }
  }
}
