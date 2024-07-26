package com.deeplearning.layer

import com.deeplearning.CostManager.{batchNormalize, dotProduct, normalize, softMax}
import com.deeplearning.{ActivationManager, ComputeInputs, ComputeWeighted, CostManager, LayerManager, Network, Normalisation}
import com.deeplearning.Network.generateRandomBiasFloat

import java.time.{Duration, Instant}

class DenseActivationLayer extends ActivationLayer {
  private val parameters = scala.collection.mutable.HashMap.empty[String,String]
  private val weightedMin = scala.collection.mutable.HashMap.empty[String,Float]
  private val weightedMax = scala.collection.mutable.HashMap.empty[String, Float]
  private val deltaTmp = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  private val weightsTmp = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  private var gamma = 0.1f
  private var beta = 0.1f
  private var ucIndex = 0

  override def ComputeZ(epoch: Int, correlationId: String, yLabel: Int, trainingCount: Int, shardedWeighted: Array[Float], internalSubLayer: Int, layer: Int, shards: Int, params : scala.collection.mutable.HashMap[String,String]): Array[Float] = {
    if (lastEpoch != epoch) {
      counterTraining = 0
      counterBackPropagation = 0
      counterFeedForward = 0
      lastEpoch = epoch
    }

    // bias initialized only one time during the training cycle
    counterTraining += 1
    if (!bInitialized) {
      bias = generateRandomBiasFloat(LayerManager.GetDenseActivationLayerStep(layer))
      bInitialized = true
      parameters += ("min" -> "0")
      parameters += ("max" -> "0")
      ucIndex = internalSubLayer
    }

    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)
      messagePropagateReceived += (correlationId -> 0)
      weightedMin += (correlationId -> 0f)
      weightedMax += (correlationId -> 0f)
    }

    if (!inProgress.contains(correlationId)) {
      inProgress += (correlationId -> true)
      shardReceived += (correlationId -> 0)
    }
    shardReceived(correlationId) += 1
/*
    val minIn = parameters("min").toFloat
    val maxIn = parameters("max").toFloat
    val min = params("min").toFloat
    val max = params("max").toFloat
    if (min < minIn) parameters("min") = params("min")
    if (max > maxIn) parameters("max") = params("max")
    val minIn2 = weightedMin(correlationId)
    val maxIn2 = weightedMax(correlationId)
    val min2 = params("weighted_min").toFloat
    val max2 = params("weighted_max").toFloat
    if (min2 < minIn2) weightedMin(correlationId) = params("weighted_min").toFloat
    if (max2 > maxIn2) weightedMax(correlationId) = params("weighted_max").toFloat
*/
    if (shardReceived(correlationId) <= shards) {
      if (!weighted.contains(correlationId))
      {
        weighted += (correlationId -> shardedWeighted)
      }
      else {
        weighted(correlationId) = CostManager.sum2(weighted(correlationId), shardedWeighted)
      }
    }

   // if (Network.debug && Network.debugLevel >= 5) context.log.info(correlationId + " " + shardReceived(correlationId).toString)
    //all received. Lets compute the activation function
    if (shards == shardReceived(correlationId) && inProgress(correlationId)) {
      //val normalizeWeights = Normalization.forward(weighted(correlationId), weighted.size)
      //val min = weighted(correlationId).min
      //val max = weighted(correlationId).max
     // val norm = batchNormalize(weighted(correlationId))

      //val mav = Normalisation.getMeanAndVariance(activation(correlationId))
      //activation(correlationId) = Normalisation.batchNormalize(activation(correlationId), mav._1, mav._3, this.gamma, this.beta)
      this.activation(correlationId) = weighted(correlationId)
      if (counterTraining % Network.minibatchBuffer == 0 && Network.debug) {
        for (j <- activation(correlationId).indices) {
          println("Activation Layer ("+layer +"): " + activation(correlationId)(j) + " " + bias(j))
        }
        println("Params : " + parameters("min") + " " + parameters("max"))
        println("--------------------------------------------------------------------")
      }

      if (Network.CheckNaN) {
        val nanIndices = activation(correlationId).zipWithIndex.filter { case (value, _) => value.isNaN || value == 0f }
        // Check if there are any NaN values
        if (nanIndices.nonEmpty) {
          println("NaN values found at indices:")
          nanIndices.foreach { case (_, index) => println(index) }
        } else {
          println("No NaN values found in the array.")
        }
      }

      if (Network.NaN) {
        activation(correlationId) = CostManager.EliminateNaN(activation(correlationId))
      }

      //context.log.info(s"${Network.Activation(layer)} on ${layer} / ${internalSubLayer} with ${activation(correlationId).length} activations : Done !")
      inProgress(correlationId) = false
      shardReceived(correlationId) = 0

      //should we propagate to next hidden layer?
      val uCs = Network.getHiddenLayersDim(layer+1, "weighted")
      for (i <- 0 until uCs) {
        val actorweightedLayer = Network.LayersIntermediateRef("weightedLayer_" + (layer+1) + "_" + i)
        actorweightedLayer ! ComputeWeighted.Weights(epoch, correlationId, yLabel, trainingCount,  activation(correlationId),uCs, i, layer+1,params)
      }

      activation(correlationId)
    }
    else
      null
  }

  override def BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Int, layer: Int, internalSubLayer: Int,params : scala.collection.mutable.HashMap[String,String]): Boolean = {
    counterBackPropagation += 1
    //compute the derivative
    //context.log.info(s"Receiving backprogation request correlationId $correlationId HiddenLayer_${layer}_${internalSubLayer}")
    messagePropagateReceived(correlationId) += 1

    if (!backPropagateReceived.contains(correlationId)) {
      backPropagateReceived += (correlationId -> true)
      deltaTmp += (correlationId -> Array.fill(delta.size)(0f))
    }
    val previousLayer = layer - 1
    var hiddenLayerStep = 0
    var fromArraySize = 0

    val nextlayer = layer + 1
    var arraySize = 0

    if (LayerManager.IsLast(nextlayer)) {
      hiddenLayerStep = LayerManager.GetOutputLayerStep()
      fromArraySize = Network.OutputLayerDim
    }
    else {
      hiddenLayerStep = LayerManager.GetDenseActivationLayerStep(layer)
      fromArraySize = Network.getHiddenLayersDim(layer, "hidden")
    }

   // val prime = ActivationManager.ComputePrime(Network.getActivationLayersType(layer), this.Z(correlationId))
/*
    if (messagePropagateReceived(correlationId) < arraySize2) {
      if (Network.debugActivity)
        println("Prime " + layer + " " + internalSubLayer)
      deltaTmp(correlationId) = CostManager.sum2(deltaTmp(correlationId), delta)
    }
    else {

 */
      //deltaTmp(correlationId) = CostManager.sum2(deltaTmp(correlationId), delta)
      minibatch(correlationId) += 1
      var deltaprime: Array[Float] = Array.fill(1)(0f)
      if (Network.debugActivity)
        println("Prime " + layer + " " + internalSubLayer)
      //if (prime.size < delta.size) {
        //val test = deltaTmp(correlationId).grouped(prime.size).toArray.transpose.map(_.sum)
        //test = CostManager.sum2(test, splitedDelta(2))
        //test = CostManager.sum2(test, splitedDelta(3))
        //test = CostManager.sum2(test, splitedDelta(4))

        //val atranspose = DenseMatrix(weighted(correlationId)).t
      //deltaTmp(correlationId) = CostManager.dotProduct(, deltaprime)
      //deltaprime = dotProduct(prime, delta)
      //}
      //else
       // deltaprime = dotProduct(prime, deltaTmp(correlationId))

      if (Network.NaN) {
        deltaprime = CostManager.EliminateNaN(deltaprime)
      }
      //deltaprime = CostManager.matMulScalar(arraySize2,deltaprime)
      nablas_b += (correlationId -> deltaprime)

      if (Network.debugActivity)
        println("Backpropagation AL (" + layer + ") " + " " + internalSubLayer)
      if (LayerManager.IsFirst(previousLayer)) {
        hiddenLayerStep = LayerManager.GetInputLayerStep()
        arraySize = Network.InputLayerDim
        for (i <- 0 until arraySize) {
          val inputLayerRef = Network.LayersInputRef("inputLayer_" + i)
          inputLayerRef ! ComputeInputs.BackPropagate(correlationId, deltaprime, learningRate, regularisation, nInputs, i, ucIndex, params)
        }
      }
      else {
        hiddenLayerStep = LayerManager.GetDenseActivationLayerStep(previousLayer)
        arraySize = Network.getHiddenLayersDim(previousLayer, "hidden")
        for (i <- 0 until arraySize) {
          if (Network.debugActivity)
            println("Send WL " + previousLayer + " " + i + " " + internalSubLayer)
          val intermediateLayerRef = Network.LayersIntermediateRef("weightedLayer_" + previousLayer + "_" + i)
          intermediateLayerRef ! ComputeWeighted.BackPropagate(correlationId, deltaprime, learningRate, regularisation, nInputs, previousLayer, i, ucIndex, params)
        }
      }
    //}

    // check if we reach the last mini-bacth
    var verticalUCs = Network.getHiddenLayersDim(layer, "hidden")

    if ((Network.MiniBatch * fromArraySize) == minibatch.values.sum) {

      parameters("min") = "0"
      parameters("max") = "0"

      //if (Network.debug && Network.debugLevel == 4) context.log.info(s"Applying gradients to bias layer $layer / $internalSubLayer")
      nablas_b.clear()
      backPropagateReceived.clear()
      weighted.clear()
      activation.clear()
      inProgress.clear()
      shardReceived.clear()
      minibatch.clear()
      Z.clear()
      weightedMin.clear()
      weightedMax.clear()
      deltaTmp.clear()
      weightsTmp.clear()
      true
    }
    else
      false
  }

  override def FeedForwardTest(correlationId: String, shardedWeighted: Array[Float], internalSubLayer: Int, layer: Int, shards: Int): Array[Float] = {
    counterFeedForward += 1
    if (layer == 2) {
      val test = 1
    }

    if (!minibatch.contains(correlationId)) {
      minibatch += (correlationId -> 0)

      var activation_tmp: Array[Float] = Array[Float]()
      var weighted_tmp: Array[Float] = Array[Float]()

      activation_tmp = Array.ofDim(LayerManager.GetDenseActivationLayerStep(layer))
      weighted_tmp = Array.fill(LayerManager.GetDenseActivationLayerStep(layer))(0)

      activation += (correlationId -> activation_tmp)
      weighted += (correlationId -> weighted_tmp)
    }

    if (!inProgress.contains(correlationId)) {
      inProgress += (correlationId -> true)
      shardReceived += (correlationId -> 0)
    }
    shardReceived(correlationId) += 1

    if (shardReceived(correlationId) <= shards) {
      if (!weighted.contains(correlationId)) {
        weighted += (correlationId -> shardedWeighted)
      }
      else {
        weighted(correlationId) = CostManager.sum2(weighted(correlationId), shardedWeighted)
      }
    }

    //all received. Lets compute the activation function
    if (shards == shardReceived(correlationId) && inProgress(correlationId)) {
      val z = CostManager.sum2(weighted(correlationId), bias)
      activation(correlationId) = ActivationManager.ComputeZ(Network.getActivationLayersType(layer), z)

      if (Network.NaN) {
        activation(correlationId) = CostManager.EliminateNaN(activation(correlationId))
      }

      inProgress(correlationId) = false
      shardReceived(correlationId) = 0

      //should we propagate to next hidden layer?
      val weightsTmp = activation(correlationId)
      //should we propagate to next hidden layer?
      val actorweightedLayer = Network.LayersIntermediateRef("weightedLayer_" + (layer+1) + "_" + internalSubLayer)
      //context.log.info("Sending activation to intermediate " + layer + "_" + internalSubLayer)
      actorweightedLayer ! ComputeWeighted.FeedForwardTest( correlationId,  activation(correlationId), internalSubLayer, layer+1)

      shardReceived -= (correlationId)
      inProgress -= (correlationId)
      activation -= (correlationId)
      weighted -= (correlationId)
      minibatch -= (correlationId)
      weightsTmp
    }
    else
      null
  }
}
