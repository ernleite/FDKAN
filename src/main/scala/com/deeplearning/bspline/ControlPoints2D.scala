package com.deeplearning.bspline

import breeze.linalg._
import breeze.stats.distributions.Rand.FixedSeed.randBasis
import breeze.stats.distributions._
import com.deeplearning.Network

import scala.util.Random
import scala.util.control.Breaks.break

object KANControlPoints {
  val rangeMin = -1.0
  val rangeMax = 1.0

  def initializeControlPoints2D(numPoints: Int = 4, minX: Double = 0.0, maxX: Double = 1.0, minY: Double = 0.0, maxY: Double = 1.0): DenseMatrix[Double] = {
    require(numPoints >= 2, "Number of points must be at least 2")

    // Create uniform distributions for X and Y coordinates
    val random = new Random()

    val uniformX =  random.between(rangeMin, rangeMax)
    val uniformY =  random.between(rangeMin, rangeMax)

    // Initialize the control points matrix
    val controlPoints = DenseMatrix.zeros[Double](numPoints, 2)

    // Generate x-coordinates in ascending order
    val xCoords = DenseVector.zeros[Double](numPoints)
    xCoords(0) = rangeMin
    xCoords(numPoints - 1) = rangeMax
    for (i <- 1 until numPoints - 1) {
      xCoords(i) =  random.between(rangeMin, rangeMax)
    }

    val sortedXCoords = DenseVector(xCoords.toArray.sorted)

    // Fill the matrix with the sorted x-coordinates and random y-coordinates
    for (i <- 0 until numPoints) {
      controlPoints(i, 0) = sortedXCoords(i)  // Sorted X coordinate
      controlPoints(i, 1) = random.between(rangeMin, rangeMax) // Random Y coordinate
    }

    controlPoints
  }

  def findClosestValueIndex(matrix: DenseMatrix[Double], value: Double): Int = {
    var closestIndex: Int = -1
    var closestDifference: Double = Double.MaxValue

    for (i <- 0 until matrix.rows) {
      val currentValue = matrix(i, 0)  // Only look at the first column (index 0)
      if (value > currentValue)
          closestIndex = i
      else
        return closestIndex
    }
    closestIndex
  }


  def expand(controlPoints: DenseMatrix[Double], searchedValue:Double): DenseMatrix[Double] = {
    val random = new Random()
    val startIndex = findClosestValueIndex(controlPoints, searchedValue)
    if (startIndex == 5) {
      val test = 1
    }

    if (startIndex + 1 < controlPoints.rows) {
      var value = 0.0
      value =  random.between(controlPoints(startIndex,0), controlPoints(startIndex+1,0))

      val value2 =  random.between(rangeMin, rangeMax)
      val newRow = DenseVector(value, value2)
      // Convert the new row to a DenseMatrix with a single row
      val matrix = controlPoints
      val updatedMatrix = DenseMatrix.zeros[Double](matrix.rows + 1, matrix.cols)
      for (i <- 0 to startIndex) {
        updatedMatrix(i, ::) := matrix(i, ::)
      }

      updatedMatrix(startIndex+1, ::) := newRow.t

      for (i <- startIndex+1 until matrix.rows) {
        updatedMatrix(i + 1, ::) := matrix(i, ::)
      }
      updatedMatrix
    }
    else
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