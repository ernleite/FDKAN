package com.deeplearning

import akka.actor.typed.Behavior
import akka.actor.typed.receptionist.{Receptionist, ServiceKey}
import akka.actor.typed.scaladsl.{AbstractBehavior, ActorContext, Behaviors}
import com.deeplearning.ComputeEpochs.SetStats
import com.deeplearning.layer.{ActivationLayer, LayerFactory}
class Activation(context: ActorContext[ComputeActivation.ActivationCommand]) extends AbstractBehavior[ComputeActivation.ActivationCommand](context) {
  import ComputeActivation._

  private var layer:ActivationLayer = _
  private var eventFF:Int = 0
  private var eventBP:Int = 0

  override def onMessage(msg: ComputeActivation.ActivationCommand): Behavior[ComputeActivation.ActivationCommand] = {
    msg match {
      case ComputeZ(epoch:Int, correlationId: String, yLabel:Int, trainingCount:Int, shardedWeighted: Array[Float], internalSubLayer:Int, layer:Int, shards: Int, params:scala.collection.mutable.HashMap[String,String], weights: Array[Float]) => {
        eventFF = eventFF + 1

        if (this.layer == null) {
          this.layer = LayerFactory.getActivationLayer(layer)
        }
        this.layer.ComputeZ(epoch,correlationId, yLabel, trainingCount, shardedWeighted, internalSubLayer, layer, shards, params)
      }
      this

      case BackPropagate(correlationId: String, delta: Array[Float], learningRate: Float, regularisation: Float, nInputs: Int, layer: Int, internalSubLayer: Int, params : scala.collection.mutable.HashMap[String,String]) =>
        eventBP = eventBP + 1
        this.layer.BackPropagate(correlationId, delta, learningRate, regularisation, nInputs, layer, internalSubLayer, params)
        this

      case FeedForwardTest(correlationId: String, shardedWeighted: Array[Float], internalSubLayer: Int, layer: Int, shards: Int) =>
        // bias initialized only one time during the training cycle
        this.layer.FeedForwardTest(correlationId,shardedWeighted,internalSubLayer,layer, shards)
        this

      case getStats(replyTo:String, actorIndex : Int) =>
        val epoch = Network.EpochsRef(replyTo)
        epoch ! SetStats(eventFF,eventBP, "activationLayer_" + actorIndex)
        this
    }
  }
}
trait ComputeActivationSerializable
object ComputeActivation {
  sealed trait ActivationCommand extends ComputeActivationSerializable
  final case class ComputeZ(Epoch:Int, CorrelationId: String, yLabel:Int, trainingCount:Int, Weighted: Array[Float], InternalSubLayer:Int, Layer:Int, Shards: Int, Params:scala.collection.mutable.HashMap[String,String], Weights: Array[Float]) extends ActivationCommand
  final case class BackPropagate(CorrelationId: String, delta: Array[Float], learningRate : Float, regularisation:Float, nInputs:Int, Layer:Int, InternalSubLayer:Int,params : scala.collection.mutable.HashMap[String,String]) extends ActivationCommand
  final case class FeedForwardTest(CorrelationId: String, Weighted: Array[Float], InternalSubLayer:Int, Layer:Int, Shards: Int) extends ActivationCommand
  final case class getStats(replyTo: String, actorIndex : Int) extends ActivationCommand

  def apply(actorId: String): Behavior[ActivationCommand] =
    Behaviors.setup { context =>
      context.system.receptionist ! Receptionist.Register(
        ServiceKey[ComputeActivation.ActivationCommand](actorId), context.self
      )
      new Activation(context)
    }
}

