package com.deeplearning

import breeze.linalg.DenseVector
import ai.djl.Device
import ai.djl.ndarray.NDArrays.dot
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.training.loss.Loss
import breeze.numerics.sqrt

object CostManager {
  def batchNormalize(input: Array[Float], epsilon: Float = 1e-5f): Array[Float] = {
    val mean = input.sum / input.length.toFloat
    val variance = input.map(x => (x - mean) * (x - mean)).sum / input.length.toFloat
    val normalized = input.map(x => (x - mean) / sqrt(variance + epsilon))
    normalized
  }
  def Compute(CostFunction: String, y:Array[Float], a: Array[Float]): Float = {
    CostFunction match {
      case "Quadratic" =>
        var c: Float = 0.0f
        for (i <- 0 until y.length) {
          val z = y(i)-a(i)
          val x = math.pow(z, 2).toFloat
          c += x
        }
        c / 2
      case "SSE" =>
        var c: Float = 0.0f
        for (i <- 0 until y.length) {
          val z = y(i) - a(i)
          val x = math.pow(z, 2).toFloat
          c += x
        }
        c / 2
    }
  }

  def Delta(trueLabels: Array[Float], prediction: Array[Float]): Array[Float] = {
    CostManager.minus(prediction,trueLabels)
  }

  def meanSquareError(predictions: Array[Float], actualValues: Array[Float]): Float = {
    val squaredErrors = predictions.zip(actualValues).map { case (prediction, actualValue) =>
      val error = prediction - actualValue
      error * error
    }
    squaredErrors.sum / squaredErrors.length
  }

  def EliminateNaN(arr: Array[Float]): Array[Float] = {
    arr.map {
      case x if x.isNaN => 0.0f
      case x => x
    }
  }
  import scala.math.log

  def categoricalCrossEntropy2(trueLabels: Array[Array[Float]], prediction: Array[Array[Float]]): Float = {
    var manager: NDManager = NDManager.newBaseManager(Device.cpu())
    if (Network.GpuMode) manager = NDManager.newBaseManager(Device.gpu(0))

    val pred: NDArray = manager.create(prediction)
    val trueLab : NDArray = manager.create(trueLabels)
    val ndList1: NDList = new NDList()
    ndList1.add(trueLab)
    val ndList2: NDList = new NDList()
    ndList2.add(pred)
    // Calculate the softmax cross-entropy loss for the minibatch
    val lossOutput = Loss.softmaxCrossEntropyLoss().evaluate(ndList1 , ndList2).toFloatArray
    ndList1.close()
    ndList2.close()
    pred.close()
    trueLab.close()
    // Don't forget to close the manager when done
    manager.close()
    lossOutput(0)
  }
  def scaleToRange(arr: Array[Float], targetMin: Float = -1f, targetMax: Float = 1f): Array[Float] = {
    if (arr.isEmpty) return arr

    val currentMin = arr.min
    val currentMax = arr.max

    // Check if all values are the same to avoid division by zero
    if (currentMin == currentMax) {
      return Array.fill(arr.length)(0f) // Map to the middle of the target range
    }

    arr.map { x =>
      val scaled = (x - currentMin) / (currentMax - currentMin)
      scaled * (targetMax - targetMin) + targetMin
    }
  }
  def normalize2(arr: Array[Float]): Array[Float] = {
    val max = arr.max
    val min = arr.min
    arr.map(v => (v - min) / (max - min))
  }

  def categoricalCrossEntropy(trueLabels: Array[Float], prediction: Array[Float]): Float = {
    if (Network.GpuMode) {
      val manager = NDManager.newBaseManager(Device.gpu(0))

      // Generate example prediction and target tensors (replace with your actual data)
      val array1 = manager.create(trueLabels)
      val fromMat1: NDArray = manager.from(array1)
      val fromList: NDList = new NDList()
      fromList.add(fromMat1)
      val array2 = manager.create(prediction)
      val fromMat2: NDArray = manager.from(array2)
      val fromPred: NDList = new NDList()
      fromPred.add(fromMat2)

      // Calculate the categorical cross-entropy loss
      val loss = Loss.softmaxCrossEntropyLoss().evaluate(fromList, fromPred).toFloatArray

      fromMat1.close()
      fromPred.close()
      fromList.close()
      fromMat2.close()
      manager.close()
      loss(0)
    }
    else {
      val manager = NDManager.newBaseManager(Device.cpu())

      // Generate example prediction and target tensors (replace with your actual data)
      val array1 = manager.create(trueLabels)
      val fromMat1: NDArray = manager.from(array1)
      val fromList: NDList = new NDList()
      fromList.add(fromMat1)
      val array2 = manager.create(prediction)
      val fromMat2: NDArray = manager.from(array2)
      val fromPred: NDList = new NDList()
      fromPred.add(fromMat2)

      // Calculate the categorical cross-entropy loss
      val loss = Loss.softmaxCrossEntropyLoss().evaluate(fromList, fromPred).toFloatArray
      fromMat1.close()
      fromPred.close()
      fromList.close()
      fromMat2.close()
      manager.close()
      loss(0)

    }
  }



  def crossEntropyLoss(predictedProbabilities: Array[Float], trueLabels: Array[Float]): Float = {
    require(predictedProbabilities.length == trueLabels.length, "Input arrays must have the same length.")

    val numClasses = predictedProbabilities.length
    var loss = 0.0f

    for (i <- 0 until numClasses) {
      // Avoid taking the log of 0 by adding a small epsilon
      val epsilon = 1e-15f
      val probability = Math.max(predictedProbabilities(i), epsilon)

      loss += trueLabels(i) * log(probability).toFloat
    }

    -loss
  }
  def replaceInf(v: DenseVector[Float], value: Float): DenseVector[Float] = {
    val mask = v.mapValues(x => x.isInfinite)
    v.mapValues(x => if (x.isInfinite) value else x)
  }

  def replaceNaN(v: Array[Float], value: Float): Array[Float] = {
    v.map(x => if (x.isNaN) value else x)
  }
  def dotProduct(mat1 : Array[Float], mat2: Array[Float]) : Array[Float] = {

    if (Network.GpuMode) {
      val manager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = manager.create(mat1)
      val array2 = manager.create(mat2)
      val fromMat1: NDArray = manager.from(array1)
      val fromMat2: NDArray = manager.from(array2)
      val c = fromMat1.mul(fromMat2).toFloatArray
      fromMat1.close()
      fromMat1.close()
      array1.close()
      array2.close()
      manager.close()
      c
    }
    else {
      val manager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = manager.create(mat1)
      val array2 = manager.create(mat2)
      val fromMat1: NDArray = manager.from(array1)
      val fromMat2: NDArray = manager.from(array2)
      val c = fromMat1.mul(fromMat2).toFloatArray
      fromMat1.close()
      fromMat1.close()
      array1.close()
      array2.close()
      manager.close()
      c
    }
  }

  def dotProduct2(mat1: Array[Float], mat2: Array[Float]): Array[Float] = {
    val manager: NDManager = NDManager.newBaseManager(Device.cpu())
    val array1 = manager.create(mat1)
    val array2 = manager.create(mat2)
    val fromMat1: NDArray = manager.from(array1)
    val fromMat2: NDArray = manager.from(array2)
    val c = dot(fromMat1,fromMat2).toFloatArray
    fromMat1.close()
    fromMat1.close()
    array1.close()
    array2.close()
    manager.close()
    c
  }

  def dotProduct3(size: Int, mat1: Array[Float], mat2: Array[Float]): Array[Array[Float]] = {
    var output = Array[Array[Float]]()
    output = Array.ofDim(mat1.length)

    if (Network.GpuMode) {
      val GpuManager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = GpuManager.create(mat1)
      val array2 = GpuManager.create(mat2)
      val fromMat1: NDArray = GpuManager.from(array1)
      val fromMat2: NDArray = GpuManager.from(array2)
      for (i <- mat1.indices) {
        output(i) =  fromMat2.mul(mat1(i)).toFloatArray
      }
      fromMat1.close()
      fromMat2.close()
      array1.close()
      array2.close()
      GpuManager.close()
    }
    else {
      output =  (DenseVector(mat2) * DenseVector(mat1).t).toArray.grouped(mat2.size).toArray
    }
    output
  }

  def dotProduct5(mat1: Array[Float], mat2: Array[Float]): Array[Float] = {
    var output = Array[Float]()

    if (Network.GpuMode) {

    }
    else {
      val manager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = manager.create(mat1)
      val array2 = manager.create(mat2)
      val fromMat1: NDArray = manager.from(array1)
      val fromMat2: NDArray = manager.from(array2)
      output = fromMat1.dot(fromMat2.transpose()).toFloatArray
      //output =  (DenseVector(mat2) * DenseVector(mat1).t).toArray
    }
    output
  }

  def matMulScalar(scalar: Float, input: Array[Float]): Array[Float] = {
    if (Network.GpuMode) {
      val manager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = manager.create(input)
      val c2 = array1.mul(scalar)
      array1.close()
      val tmp = c2.toFloatArray
      c2.close()
      manager.close()
      tmp
    }
    else {
      val manager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = manager.create(input)
      val c2 = array1.mul(scalar)
      array1.close()
      val tmp = c2.toFloatArray
      c2.close()
      manager.close()
      tmp
    }
  }

  def matMulScalar2(scalar: Float, input: Array[Array[Float]]): Array[Float] = {
    if (Network.GpuMode) {
      val manager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = manager.create(input)
      val c2 = array1.mul(scalar)
      array1.close()
      val tmp = c2.toFloatArray
      c2.close()
      manager.close()
      tmp
    }
    else {
      val manager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = manager.create(input)
      val c2 = array1.mul(scalar)
      array1.close()
      val tmp = c2.toFloatArray
      c2.close()
      manager.close()
      tmp
    }
  }

  def matMul3(mat1: Array[Float], mat2: Array[Float]): Array[Float] = {
    if (Network.GpuMode) {
      val cpuManager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = cpuManager.create(mat1)
      val array2 = cpuManager.create(mat2)
      val fromMat1: NDArray = cpuManager.from(array1)
      val fromMat2: NDArray = cpuManager.from(array2)
      val c2 = fromMat2.matMul(fromMat1)
      fromMat2.close()
      fromMat1.close()
      array1.close()
      val tmp = c2.toFloatArray
      c2.close()
      cpuManager.close()
      tmp
    }
    else {
      val cpuManager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = cpuManager.create(mat1)
      val array2 = cpuManager.create(mat2)
      val fromMat1: NDArray = cpuManager.from(array1)
      val fromMat2: NDArray = cpuManager.from(array2)
      val c2 = fromMat2.matMul(fromMat1)
      fromMat2.close()
      fromMat1.close()
      array1.close()
      val tmp = c2.toFloatArray
      c2.close()
      cpuManager.close()
      tmp
    }
  }

  def matMul(mat1:Array[Array[Float]], mat2:Array[Float]) : Array[Float] = {
    if (Network.GpuMode) {
      val cpuManager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = cpuManager.create(mat1)
      val array2 = cpuManager.create(mat2)
      val fromMat1: NDArray = cpuManager.from(array1)
      val fromMat2: NDArray = cpuManager.from(array2)
      val c2 = fromMat2.matMul(fromMat1)
      fromMat2.close()
      fromMat1.close()
      array1.close()
      val tmp = c2.toFloatArray
      c2.close()
      cpuManager.close()
      tmp
    }
    else {
      val cpuManager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = cpuManager.create(mat1)
      val array2 = cpuManager.create(mat2)
      val fromMat1: NDArray = cpuManager.from(array1)
      val fromMat2: NDArray = cpuManager.from(array2)
      val c2 = fromMat2.matMul(fromMat1)
      fromMat2.close()
      fromMat1.close()
      array1.close()
      val tmp =  c2.toFloatArray
      c2.close()
      cpuManager.close()
      tmp
    }
  }

  def dotProduct4(mat1: Array[Array[Float]], mat2: Array[Float]): Array[Float] = {
    val bT = mat1.transpose
    val test = mat1.toArray

    if (Network.GpuMode) {
      val GpuManager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = GpuManager.create(bT)
      val array2 = GpuManager.create(mat2)
      val fromMat1: NDArray = GpuManager.from(array1)
      val fromMat2: NDArray = GpuManager.from(array2)

      val c = fromMat1.matMul(fromMat2).toFloatArray
      fromMat2.close()
      fromMat1.close()
      array1.close()
      array2.close()
      GpuManager.close()
      c
    } else {
      val cpuManager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = cpuManager.create(bT)
      val array2 = cpuManager.create(mat2)
      val fromMat1: NDArray = cpuManager.from(array1)
      val fromMat2: NDArray = cpuManager.from(array2)

      val c = fromMat1.matMul(fromMat2).toFloatArray
      fromMat2.close()
      fromMat1.close()
      array1.close()
      array2.close()
      cpuManager.close()
      c
      //val c: Array[Float] = bT.map(row => row.zip(mat2).map { case (x, y) => x * y }.sum)
      //c
    }
  }

  def dotProduct6(size: Int, mat1: Array[Array[Float]], mat2: Array[Float]): Array[Float] = {
    var output = Array[Float]()
    output = Array.fill[Float](size)(0.0f)
    // calculate dot product of a with each row of bT
    if (Network.GpuMode) {
      val GpuManager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = GpuManager.create(mat1)
      val array2 = GpuManager.create(mat2)
      val fromMat1: NDArray = GpuManager.from(array1)
      val fromMat2: NDArray = GpuManager.from(array2)

      val c = fromMat1.matMul(fromMat2).toFloatArray
      fromMat2.close()
      fromMat1.close()
      array1.close()
      array2.close()
      GpuManager.close()
      c
    }
    else {

      val cpuManager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = cpuManager.create(mat1)
      val array2 = cpuManager.create(mat2)
      val fromMat1: NDArray = cpuManager.from(array1)
      val fromMat2: NDArray = cpuManager.from(array2)

      val c2 = fromMat1.matMul(fromMat2).toFloatArray
      fromMat2.close()
      fromMat1.close()
      array1.close()
      array2.close()
      cpuManager.close()
      c2
      /*
      val matrix: DenseMatrix[Float] = DenseMatrix.create[Float](mat1.length, mat1(0).length, mat1.flatten)
      val kernel = DenseMatrix(mat2)
      //val dot = dotProduct(inp1,deltaFilters(i))
      val dot =  matrix * kernel.t
        dot.toArray
     val f = dot.t.toArray
      val c: Array[Float] = mat1.map(row => row.zip(mat2).map { case (x, y) => x * y }.sum)
      c
*/

    }
  }

  def minusScalar(mat1: Array[Float], scalar: Float): Array[Float] = {
    if (Network.GpuMode) {
      val GpuManager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = GpuManager.create(mat1)
      val fromMat1: NDArray = GpuManager.from(array1)

      val c = fromMat1.subi(scalar).toFloatArray
      fromMat1.close()
      array1.close()
      GpuManager.close()
      c
    }
    else {
      val GpuManager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = GpuManager.create(mat1)
      val fromMat1: NDArray = GpuManager.from(array1)

      val c = fromMat1.subi(scalar).toFloatArray
      fromMat1.close()
      array1.close()
      GpuManager.close()
      c
    }
  }

  def minus2(mat1: Array[Float], mat2: Array[Float]): Array[Float] = {
    if (Network.GpuMode) {
      val manager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = manager.create(mat1)
      val array2 = manager.create(mat2)
      val fromMat1: NDArray = manager.from(array1)
      val fromMat2: NDArray = manager.from(array2)

      val c = fromMat1.sub(fromMat2).toFloatArray
      fromMat1.close()
      fromMat2.close()
      array1.close()
      array2.close()
      manager.close()
      c
    }
    else {
      val manager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = manager.create(mat1)
      val array2 = manager.create(mat2)
      val fromMat1: NDArray = manager.from(array1)
      val fromMat2: NDArray = manager.from(array2)

      val c = fromMat1.sub(fromMat2).toFloatArray
      fromMat1.close()
      fromMat2.close()
      array1.close()
      array2.close()
      manager.close()
      c
    }
  }


  def minus2(mat1: Array[Array[Float]], mat2: Array[Float]): Array[Float] = {
    if (Network.GpuMode) {
      val manager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = manager.create(mat1)
      val array2 = manager.create(mat2)
      val fromMat1: NDArray = manager.from(array1)
      val fromMat2: NDArray = manager.from(array2)

      val c = fromMat1.sub(fromMat2).toFloatArray
      fromMat1.close()
      fromMat2.close()
      array1.close()
      array2.close()
      manager.close()
      c
    }
    else {
      val manager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = manager.create(mat1)
      val array2 = manager.create(mat2)
      val fromMat1: NDArray = manager.from(array1)
      val fromMat2: NDArray = manager.from(array2)

      val c = fromMat1.sub(fromMat2).toFloatArray
      fromMat1.close()
      fromMat2.close()
      array1.close()
      array2.close()
      manager.close()
      c
    }
  }

  def minus(mat1: Array[Float], mat2: Array[Float]): Array[Float] = {
    if (Network.GpuMode) {
      val manager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = manager.create(mat1)
      val array2 = manager.create(mat2)
      val fromMat1: NDArray = manager.from(array1)
      val fromMat2: NDArray = manager.from(array2)

      val c = fromMat1.subi(fromMat2).toFloatArray
      fromMat1.close()
      fromMat2.close()
      array1.close()
      array2.close()
      manager.close()
      c
    }
    else {
      val manager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = manager.create(mat1)
      val array2 = manager.create(mat2)
      val fromMat1: NDArray = manager.from(array1)
      val fromMat2: NDArray = manager.from(array2)

      val c = fromMat1.subi(fromMat2).toFloatArray
      fromMat1.close()
      fromMat2.close()
      array1.close()
      array2.close()
      manager.close()
      c
    }
  }

  def sum(mat1 : Array[Float], scalar: Float): Array[Float] = {
    if (Network.GpuMode) {
      val GpuManager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = GpuManager.create(mat1)
      val fromMat1: NDArray = GpuManager.from(array1)

      val c = fromMat1.addi(scalar).toFloatArray
      fromMat1.close()
      array1.close()
      GpuManager.close()
      c
    }
    else {
      val GpuManager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = GpuManager.create(mat1)
      val fromMat1: NDArray = GpuManager.from(array1)

      val c = fromMat1.addi(scalar).toFloatArray
      fromMat1.close()
      array1.close()
      GpuManager.close()
      c
    }
  }

  def divide(mat1: Array[Float], scalar: Float): Array[Float] = {
    if (Network.GpuMode) {
      val GpuManager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = GpuManager.create(mat1)
      val fromMat1: NDArray = GpuManager.from(array1)

      val c = fromMat1.divi(scalar).toFloatArray
      fromMat1.close()
      array1.close()
      GpuManager.close()
      c
    }
    else {
      val CpuManager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = CpuManager.create(mat1)
      val fromMat1: NDArray = CpuManager.from(array1)

      val c = fromMat1.divi(scalar).toFloatArray
      fromMat1.close()
      array1.close()
      CpuManager.close()
      c
    }
  }

  def sum2(mat1: Array[Float], mat2: Array[Float]): Array[Float] = {
    if (Network.GpuMode) {
      val GpuManager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = GpuManager.create(mat1)
      val array2 = GpuManager.create(mat2)
      val fromMat1: NDArray = GpuManager.from(array1)
      val fromMat2: NDArray = GpuManager.from(array2)

      val c = fromMat1.addi(fromMat2).toFloatArray
      fromMat1.close()
      array1.close()
      GpuManager.close()
      c
    }
    else {
      val CpuManager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = CpuManager.create(mat1)
      val array2 = CpuManager.create(mat2)
      val fromMat1: NDArray = CpuManager.from(array1)
      val fromMat2: NDArray = CpuManager.from(array2)

      val c = fromMat1.addi(fromMat2).toFloatArray
      fromMat1.close()
      array1.close()
      CpuManager.close()
      c
    }
  }

  def sum3(mat1: Array[Array[Float]]): Array[Float] = {
    if (Network.GpuMode) {
      val GpuManager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = GpuManager.create(mat1)
      val fromMat1: NDArray = GpuManager.from(array1)

      val c = fromMat1.transpose.sum().toFloatArray
      fromMat1.close()
      array1.close()
      GpuManager.close()
      c
    }
    else {
      val CpuManager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = CpuManager.create(mat1)
      val fromMat1: NDArray = CpuManager.from(array1)
      val c = fromMat1.transpose.sum().toFloatArray

      fromMat1.close()
      array1.close()
      CpuManager.close()
      c
    }
  }

  def scalling(array: Array[Float], min: Float, max: Float, rangeX: Float, rangeY: Float): Array[Float] = {
    val scale = (rangeY - rangeX) / (max - min)
    val b = rangeX - scale * min
    array.map(_ * scale + b)
  }


  def softMax(mat:Array[Float]) : Array[Float] = {
    if (Network.GpuMode) {
      val gpuManager: NDManager = NDManager.newBaseManager(Device.gpu(0))
      val array1 = gpuManager.create(mat)
      val fromMat1: NDArray = gpuManager.from(array1)

      val c = fromMat1.softmax(0).toFloatArray
      fromMat1.close()
      array1.close()
      gpuManager.close()
      c
    }
    else {
      val cpuManager: NDManager = NDManager.newBaseManager(Device.cpu())
      val array1 = cpuManager.create(mat)
      val fromMat1: NDArray = cpuManager.from(array1)

      val c = fromMat1.softmax(0).toFloatArray
      fromMat1.close()
      array1.close()
      cpuManager.close()
      c
    }
  }
}
