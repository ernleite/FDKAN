package com.deeplearning

import akka.actor.typed.{ActorRef, Behavior}
import akka.actor.typed.receptionist.{Receptionist, ServiceKey}
import akka.actor.typed.scaladsl.{AbstractBehavior, ActorContext, Behaviors}
import breeze.linalg.{DenseVector, normalize}
import com.deeplearning.ComputeEpochs.SetStats
import com.deeplearning.CostManager.{normalize2, scaleToRange}

import java.time.{Duration, Instant}

class Output(context: ActorContext[ComputeOutput.OutputCommand]) extends AbstractBehavior[ComputeOutput.OutputCommand](context) {
  import ComputeOutput._

  private val minibatch = scala.collection.mutable.HashMap.empty[String, Boolean]
  private var bInitialized:Boolean = false
  private val shardReceived = scala.collection.mutable.HashMap.empty[String, Int]
  private val backPropagateReceived = scala.collection.mutable.HashMap.empty[String, Boolean]
  private val inProgress = scala.collection.mutable.HashMap.empty[String, Boolean]

  private val weighted = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  private var bias: Array[Float] = Array[Float]()
  private val activation = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  private val z = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  private val nablas_b = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  private var nabla_b = Array[Float]()
  private var counterTraining: Int = 0
  private var counterBackPropagation: Int = 0
  private var counterFeedForward: Int = 0
  private val debugDelta = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  private var lastEpoch = 0
  private var mse:Array[Float] = Array[Float]()
  private val trueLabels = scala.collection.mutable.HashMap.empty[String, Array[Float]]
  private val parameters = scala.collection.mutable.HashMap.empty[String,String]
  private var gamma = 0.1f
  private var beta = 0.1f
  var parameterSended = false
  private val weightedMin = scala.collection.mutable.HashMap.empty[String, Float]
  private val weightedMax = scala.collection.mutable.HashMap.empty[String, Float]
  private val weightsTmp = scala.collection.mutable.HashMap.empty[String, Array[Float]]

  private var eventFF: Int = 0
  private var eventBP: Int = 0

  override def onMessage(msg: ComputeOutput.OutputCommand): Behavior[ComputeOutput.OutputCommand] = {
    msg match {
      case Compute(epoch: Int, correlationId: String, yLabel:Int, trainingCount:Int,shardedWeighted: Array[Float], internalSubLayer:Int, layer:Int, shards: Int, params:scala.collection.mutable.HashMap[String,String]) =>
        eventFF = eventFF + 1

        if (lastEpoch != epoch) {
          counterTraining = 0
          counterBackPropagation = 0
          counterFeedForward = 0
          lastEpoch = epoch
          mse =  Array.empty[Float]
        }

        if (!bInitialized) {
          //bias = Array.ofDim(LayerManager.GetOutputLayerStep())
          //bias = generateRandomBiasFloat(LayerManager.GetOutputLayerStep())
          bInitialized = true
          parameters += ("min" -> "0")
          parameters += ("max" -> "0")
          parameters += ("weighted_min" -> "0")
          parameters += ("weighted_max" -> "0")
        }

        if (! minibatch.contains(correlationId)) {
          minibatch += (correlationId -> false)
          var activation_tmp: Array[Float] = Array[Float]()
          var weighted_tmp: Array[Float] = Array[Float]()
          var z_tmp: Array[Float] = Array[Float]()
          var nabla_b_tmp: Array[Float] = Array[Float]()

          activation_tmp = Array.ofDim(LayerManager.GetOutputLayerStep())
          weighted_tmp = Array.ofDim(LayerManager.GetOutputLayerStep())
          z_tmp = Array.ofDim(LayerManager.GetOutputLayerStep())
          nabla_b_tmp = Array.ofDim(LayerManager.GetOutputLayerStep())
          //weightedMin += (correlationId -> 0f)
          //weightedMax += (correlationId -> 0f)
          activation += (correlationId -> activation_tmp)
          //z += (correlationId -> z_tmp)
          weighted += (correlationId -> weighted_tmp)
          nablas_b += (correlationId -> nabla_b_tmp)
          nabla_b = Array.ofDim(LayerManager.GetOutputLayerStep())
        }

        if (!inProgress.contains(correlationId)) {
          inProgress += (correlationId -> true)
          shardReceived += (correlationId -> 0)
        }

        shardReceived(correlationId) += 1

        if (shardReceived(correlationId) <= shards) {
          weighted(correlationId) = CostManager.sum2(weighted(correlationId), shardedWeighted)
        }

        //all received. Lets compute the activation function
        if (shards == shardReceived(correlationId) && inProgress(correlationId)) {
          counterTraining +=1
          this.activation(correlationId) = weighted(correlationId)
          activation(correlationId) = scaleToRange(activation(correlationId))
          activation(correlationId) = ActivationManager.ComputeZ(Network.OutputActivationType, activation(correlationId))

          if (counterTraining % Network.minibatchBuffer == 0) {
            context.log.info("--------------------------------------------------------------------")
            context.log.info("Label src : " + yLabel)

            for (j <- activation(correlationId).indices) {
              context.log.info("Output Layer: " + this.activation(correlationId)(j))
            }
            context.log.info("--------------------------------------------------------------------")
          }

          inProgress(correlationId) = false
          shardReceived(correlationId) = 0
          var arr = Array[Float]()
          arr = Array.fill[Float](Network.OutputLayer)(0.0f)
          arr(yLabel) = 1
          var tLabels = Array[Float]()
          tLabels = Array.ofDim(1)
          tLabels(0) = yLabel.toFloat
          trueLabels += (correlationId -> tLabels)
         // context.log.info(cost.toString)
          context.self! ComputeLoss(correlationId, arr, trainingCount, Network.LearningRate, Network.Regularisation, internalSubLayer, params)
        }
        this

      case ComputeLoss(correlationId: String, labels:Array[Float], trainingLabelsCount:Int, learningRate:Float, regularisation:Float, internalSubLayer:Int, params : scala.collection.mutable.HashMap[String,String]) =>
        //context.log.info(s"Label is : $searchedLabel in trainingSet ${trainingLabelsCount} for correlationId $correlationId")
        val delta = CostManager.Delta(labels,activation(correlationId))
        if (counterTraining % Network.minibatchBuffer == 9999) {
          context.log.info("--------------------------------------------------------------------")
          for (j <- delta.indices) {
            context.log.info("Output Layer delta: " + delta(j))
          }
          context.log.info("--------------------------------------------------------------------")
        }

        //nablas_b += (correlationId->delta)
        context.self! BackPropagate(correlationId, delta , learningRate, regularisation, trainingLabelsCount, Network.HiddenLayers.length*2, internalSubLayer, params)
        this

      case BackPropagate(correlationId:String, delta : Array[Float], learningRate: Float, regularisation: Float, nInputs:Int, layer: Int, internalSubLayer:Int, params : scala.collection.mutable.HashMap[String,String])  =>
        eventBP = eventBP + 1

        counterBackPropagation +=1
        val startedAt = Instant.now

        //compute the derivative
        //We need to calculate how many sublayers we need to call
        val intermediateLayers = Network.getHiddenLayersDim(layer, "weighted")
        if (!backPropagateReceived.contains(correlationId)) {
          backPropagateReceived += (correlationId -> true)
        }

        /*
        Dot Product Weights
         */
        for (i: Int <- 0 until intermediateLayers) {
          // context.log.info(s"$correlationId Backpropagate from Output $layer $i")
          val intermediateLayerRef = Network.LayersIntermediateRef("weightedLayer_" + layer + "_" + i)
          intermediateLayerRef ! ComputeWeighted.BackPropagate(correlationId, delta, learningRate, regularisation, nInputs, layer, i, internalSubLayer, params)
        }

        if (backPropagateReceived.size == Network.MiniBatch) {
          //we need to calculate the loss function
          val trueLabs = trueLabels.values.toArray.flatten.grouped(1).toArray
          val predictions = activation.values.toArray.flatten.grouped(Network.OutputLayer).toArray
          val cost = CostManager.categoricalCrossEntropy2(trueLabs,predictions)
          mse = mse :+ cost
          /*
            self.biases = [b-(eta/len(mini_batch))*nb
                          for b, nb in zip(self.biases, nabla_b)]
           */
          parameterSended = false
          parameters("min") = "0"
          parameters("max") =  "0"
          //gamma -= delta.sum / delta.length
          //beta -= delta.sum/ delta.length
          //println("Backpropagation : Output")
          z.clear()
          trueLabels.clear()
          shardReceived.clear()
          inProgress.clear()
          nablas_b.clear()
          activation.clear()
          weighted.clear()
          minibatch.clear()
          backPropagateReceived.clear()
          weightedMin.clear()
          weightedMax.clear()
          weightsTmp.clear()

          val endedAt = Instant.now
          val duration = Duration.between(startedAt, endedAt).toMillis

          if (counterTraining % Network.minibatchBuffer == 0 && duration>1) {
            println("-------------------------------------------")
            println("Ouput backward duration : " + duration)
          }
        }
        this

      case FeedForwardTest(correlationId: String, shardedWeighted: Array[Float], internalSubLayer: Int, layer: Int, shards: Int) =>
        counterFeedForward+=1

        if (!minibatch.contains(correlationId)) {
          minibatch += (correlationId -> false)
          var activation_tmp: Array[Float] = Array[Float]()
          var weighted_tmp: Array[Float] = Array[Float]()
          var z_tmp: Array[Float] = Array[Float]()

          activation_tmp = Array.ofDim(LayerManager.GetOutputLayerStep())
          weighted_tmp = Array.ofDim(LayerManager.GetOutputLayerStep())
          z_tmp = Array.ofDim(LayerManager.GetOutputLayerStep())

          activation += (correlationId -> activation_tmp)
          weighted += (correlationId -> weighted_tmp)
        }

        if (!inProgress.contains(correlationId)) {
          inProgress += (correlationId -> true)
          shardReceived += (correlationId -> 0)
        }
        shardReceived(correlationId) += 1

        if (shardReceived(correlationId) <= shards) {
          weighted(correlationId) = CostManager.sum2(weighted(correlationId), shardedWeighted)
        }

        //all received. Lets compute the activation function
        if (shards == shardReceived(correlationId) && inProgress(correlationId)) {

          inProgress(correlationId) = false
          shardReceived(correlationId) = 0
          this.activation(correlationId) = weighted(correlationId)
          //activation(correlationId) = normalize2(weighted(correlationId))
          //scaleToRange(activation(correlationId))
          this.activation(correlationId) = scaleToRange(activation(correlationId))
          val maxlabel =  activation(correlationId).maxBy(t => t.self)
          val label =  activation(correlationId).indexOf(maxlabel)

          val epoch = Network.EpochsRef("epoch_0")
          epoch ! ComputeEpochs.NotifyFeedForwardTest(correlationId, label, "outputLayer_" + internalSubLayer)

          activation -= (correlationId)
          shardReceived -= (correlationId)
          weighted -= (correlationId)
          minibatch -= (correlationId)
          inProgress -= (correlationId)
          nablas_b -= (correlationId)
        }
        this

      case ComputeEvaluate(correlationId: String, labels: Array[Float]) =>
        val searchedLabel = labels.indexOf(1)
        context.log.info("")
        context.log.info(s"--------------------------------------------------")
        context.log.info(s"Searched Label !")
        context.log.info(s"--------------------------------------------------")
        context.log.info("")
        this

      case GetErrorRate(replyTo:ActorRef[ComputeEpochs.ErrorRate], params:scala.collection.mutable.HashMap[String,String]) =>
        val mseMean = mse.sum / mse.length

        replyTo! com.deeplearning.ComputeEpochs.ErrorRate(mseMean, params)
        this

      case getStats(replyTo:String, actorIndex : Int) =>
        val epoch = Network.EpochsRef(replyTo)
        epoch ! SetStats(eventFF,eventBP, "outputLayer_"+actorIndex)
        this
    }
  }
}
trait ComputeOutputSerializable
object ComputeOutput {
  sealed trait OutputCommand extends ComputeOutputSerializable
  final case class Compute(Epoch:Int, correlationId: String, yLabel:Int, trainingCount:Int, Weighted: Array[Float], InternalSubLayer:Int, Layer:Int, Shards: Int, Params:scala.collection.mutable.HashMap[String,String]) extends OutputCommand
  final case class ComputeLoss(CorrelationId: String, Label:Array[Float], TrainingLabelsCount: Int, LearningRate: Float, regularisation: Float, InternalSubLayer:Int,Params:scala.collection.mutable.HashMap[String,String]) extends OutputCommand
  final case class ComputeEvaluate(CorrelationId: String, Label:Array[Float]) extends OutputCommand
  final case class GetErrorRate(replyTo:ActorRef[ComputeEpochs.ErrorRate], Params:scala.collection.mutable.HashMap[String,String]) extends OutputCommand
  final case class BackPropagate(CorrelationId:String, Delta : Array[Float],LearningRate: Float, Regularisation: Float, nInputs:Int, Layer: Int,InternalSubLayer:Int,Params:scala.collection.mutable.HashMap[String,String])  extends  OutputCommand
  final case class FeedForwardTest(correlationId: String, Weighted: Array[Float], InternalSubLayer:Int, Layer:Int, Shards: Int) extends OutputCommand
  final case class getStats(replyTo: String, actorIndex : Int) extends OutputCommand

  def apply(actorId: String): Behavior[OutputCommand] =
    Behaviors.setup { context =>
      context.system.receptionist ! Receptionist.Register(
        ServiceKey[ComputeOutput.OutputCommand](actorId), context.self
      )
      new Output(context)
    }
}