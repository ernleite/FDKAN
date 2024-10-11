package com.deeplearning.bspline

import breeze.linalg.{DenseMatrix, DenseVector}
import com.deeplearning.bspline.BSpline2.computeBSplines

import scala.collection.mutable.ArrayBuffer

object BSpline2 {

  def linear(input: DenseVector[Double], weight: DenseMatrix[Double], bias: Option[DenseVector[Double]] = None): DenseVector[Double] = {
    // Perform the matrix-vector multiplication
    val result = weight * input

    // Add the bias if it exists
    bias match {
      case Some(b) => result + b
      case None => result
    }
  }

  def computeBSplines(x: DenseMatrix[Double], grid: DenseMatrix[Double], inFeatures: Int, gridSize: Int, splineOrder: Int): DenseMatrix[Double] = {
    require(x.cols == inFeatures, "Input tensor must have the correct number of features.")

    // Print the shape of x
    println(s"x shape: (${x.rows}, ${x.cols})")

    // Unsqueeze x to add an extra dimension (simulate by using a 3D array)
    val xUnsqueezed = Array.tabulate(x.rows, x.cols, 1)((i, j, _) => x(i, j))
    println(s"x unsqueezed shape: (${xUnsqueezed.length}, ${xUnsqueezed(0).length}, ${xUnsqueezed(0)(0).length})")

    // Initialize bases
    val bases = Array.ofDim[Double](x.rows, inFeatures, gridSize + splineOrder)

    // Compute the initial bases
    for (i <- 0 until x.rows; j <- 0 until inFeatures) {
      for (k <- 0 until gridSize + splineOrder) {
        if (xUnsqueezed(i)(j)(0) >= grid(j, k) && xUnsqueezed(i)(j)(0) < grid(j, k + 1)) {
          bases(i)(j)(k) = 1.0
        } else {
          bases(i)(j)(k) = 0.0
        }
      }
    }

    // Compute the B-spline bases
    for (k <- 1 to splineOrder) {
      for (i <- 0 until x.rows; j <- 0 until inFeatures) {
        for (m <- 0 until gridSize + splineOrder - k) {
          val left = (xUnsqueezed(i)(j)(0) - grid(j, m)) / (grid(j, m + k) - grid(j, m))
          val right = (grid(j, m + k + 1) - xUnsqueezed(i)(j)(0)) / (grid(j, m + k + 1) - grid(j, m + 1))
          bases(i)(j)(m) = left * bases(i)(j)(m) + right * bases(i)(j)(m + 1)
        }
      }
    }

    // Convert the 3D array back to a DenseMatrix for output (flattening the last two dimensions)
    val flattenedBases = bases.map(_.flatten)
    DenseMatrix(flattenedBases: _*)
  }

  def curve2coeff(x: DenseMatrix[Double], y: DenseMatrix[Double], inFeatures: Int, outFeatures: Int, gridSize: Int, splineOrder: Int): DenseMatrix[Double] = {
    require(x.cols == inFeatures, "Input tensor x must have the correct number of features.")
    require(y.rows == x.rows && y.cols == inFeatures * outFeatures, "Output tensor y must have the correct shape.")

    // Simulate b_splines function
    val A = bSplines(x, inFeatures, gridSize, splineOrder) // (batch_size, grid_size + spline_order)
    val B = y

    println("----------------------A--------------------------")
    println(A)
    println("----------------------B--------------------------")
    println(B)

    // Solve the least squares problem
    val solution = A \ B

    // Permute dimensions to match the expected output shape
    val result = permuteSolution(solution, outFeatures, inFeatures, gridSize, splineOrder)

    require(result.rows == outFeatures && result.cols == inFeatures * (gridSize + splineOrder), "Result must have the correct shape.")
    result
  }

  def bSplines(x: DenseMatrix[Double], inFeatures: Int, gridSize: Int, splineOrder: Int): DenseMatrix[Double] = {
    // Placeholder for b_splines logic
    // Assuming it returns a matrix of shape (batch_size, grid_size + spline_order)
    DenseMatrix.rand[Double](x.rows, gridSize + splineOrder)
  }

  def permuteSolution(solution: DenseMatrix[Double], outFeatures: Int, inFeatures: Int, gridSize: Int, splineOrder: Int): DenseMatrix[Double] = {
    // Reshape solution to simulate 3D reshaping using a 2D matrix
    val reshapedRows = inFeatures * (gridSize + splineOrder)
    val reshapedCols = outFeatures
    val reshaped = DenseMatrix.zeros[Double](reshapedRows, reshapedCols)

    for (i <- 0 until inFeatures; j <- 0 until gridSize + splineOrder; k <- 0 until outFeatures) {
      reshaped(i * (gridSize + splineOrder) + j, k) = solution(j, i * outFeatures + k)
    }

    // Create a permuted result matrix
    val permuted = DenseMatrix.zeros[Double](outFeatures, inFeatures * (gridSize + splineOrder))
    for (i <- 0 until outFeatures; j <- 0 until inFeatures; k <- 0 until gridSize + splineOrder) {
      permuted(i, j * (gridSize + splineOrder) + k) = reshaped(j * (gridSize + splineOrder) + k, i)
    }
    permuted
  }

}

// Example usage
object BSplineExample {
  def main(args: Array[String]): Unit = {
    // Example values
    val batchSize = 4
    val inFeatures = 3
    val gridSize = 5
    val splineOrder = 2

    // Ensure x has the correct shape (batchSize, inFeatures)
    val x = DenseMatrix.rand[Double](batchSize, inFeatures)

    // Ensure grid has the correct shape (inFeatures, gridSize + 2 * splineOrder + 1)
    val grid = DenseMatrix.rand[Double](inFeatures, gridSize + 2 * splineOrder + 1)

    val bSplines = computeBSplines(x, grid, inFeatures, gridSize, splineOrder)
    println("B-spline bases:")
    println(bSplines)
  }
}
