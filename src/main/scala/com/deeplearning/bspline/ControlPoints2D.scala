package com.deeplearning.bspline

import breeze.linalg._
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import breeze.stats.distributions._
import com.deeplearning.Network

import scala.util.Random

object KANControlPoints {

  def initializeControlPoints2D(numPoints: Int = 4, minX: Double = 0.0, maxX: Double = 1.0, minY: Double = 0.0, maxY: Double = 1.0): DenseMatrix[Double] = {
    require(numPoints >= 2, "Number of points must be at least 2")

    // Create uniform distributions for X and Y coordinates
    val random = new Random()

    val uniformX =  random.between(0.0, 10.0)
    val uniformY =  random.between(-10.0, 10.0)

    // Initialize the control points matrix
    val controlPoints = DenseMatrix.zeros[Double](numPoints, 2)

    // Generate x-coordinates in ascending order
    val xCoords = DenseVector.zeros[Double](numPoints)
    xCoords(0) = minX
    xCoords(numPoints - 1) = maxX
    for (i <- 1 until numPoints - 1) {
      xCoords(i) =  random.between(0.0, 10.0)
    }
    val sortedXCoords = DenseVector(xCoords.toArray.sorted)

    // Fill the matrix with the sorted x-coordinates and random y-coordinates
    for (i <- 0 until numPoints) {
      controlPoints(i, 0) = sortedXCoords(i)  // Sorted X coordinate
      controlPoints(i, 1) = random.between(-10.0, 10.0) // Random Y coordinate
    }

    controlPoints
  }

  def main(args: Array[String]): Unit = {
    val k = 3
    val minControlPoints = k + 1
    val controlPoints = initializeControlPoints2D(minControlPoints)

    println(s"Initialized Control Points for k=$k:")
    println(controlPoints)
  }
}