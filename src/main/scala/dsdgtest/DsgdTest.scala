/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package dsdgtest

import org.apache.flink.api.scala._
import org.apache.flink.api.java.utils.ParameterTool
import org.apache.flink.ml.evaluation.RankingRecommendationScores
import org.apache.flink.ml.recommendation.SGD

import scala.language.postfixOps
import scala.util.Random

object DsgdTest {

  def main(args: Array[String]): Unit = {

    val params = ParameterTool.fromArgs(args)

    // set up execution environment
    val env = ExecutionEnvironment.getExecutionEnvironment

    val inputPath = params.get("inputPath")
    val outputPath = params.get("outputPath")
    val properties = params.get("properties")

    def getPredictions(iterations: Int, lambda: Int, numBlocks: Int, numFactors: Int, learningRate: Double,
                seed: Long, trainDS: DataSet[(Int, Int, Double)], testDS: DataSet[(Int, Int, Double)]) = {

      val dsgd = SGD()
        .setIterations(iterations)
        .setLambda(lambda)
        .setBlocks(numBlocks)
        .setNumFactors(numFactors)
        .setLearningRate(learningRate)
        .setSeed(seed)

      dsgd.fit(trainDS)

      val testWithoutRatings = testDS.map(i => (i._1, i._2))

      val predDS = dsgd.predict(testWithoutRatings)
      predDS
    }

    def getRmse(testDS: DataSet[(Int, Int, Double)], predDS: DataSet[(Int, Int, Double)]) = {

      val rmse = testDS.join(predDS).where(0, 1).equalTo(0, 1)
        .map(i => (i._1._3, i._2._3))
        .map(i => (i._1 - i._2) * (i._1 - i._2))
        .map(i => (i, 1))
        .reduce((i, j) => (i._1 + j._1, i._2 + j._2))
        .map(i => math.sqrt(i._1 / i._2))

      rmse.collect().head
    }

    val trainPath = inputPath + "_train.csv"
    val testPath = inputPath + "_test.csv"


    val trainDS: DataSet[(Int, Int, Double)] = env.readCsvFile[(Int, Int, Double)](trainPath)
    val testDS = env.readCsvFile[(Int, Int, Double)](testPath)

    val props = scala.io.Source.fromFile(properties).getLines
      .map(line => (line.split(":")(0), line.split(":")(1).split(",").toList)).toMap

    val iterations = props("iterations").map(_.toInt)
    val blocks = props("blocks").map(_.toInt)
    val dimension = props("dimension").map(_.toInt)
    val learningRate = props("learningrate").map(_.toDouble)

    for (i <- iterations) {
      for (b <- blocks) {
        for (lr <- learningRate) {
          for (d <- dimension) {
            val seed = Random.nextLong()
            val predDS = getPredictions(i, 0, b, d, lr, seed, trainDS, testDS)
            val rmse = getRmse(testDS, predDS)
            val result = s"$i,$b,$lr,$rmse\n"
            scala.tools.nsc.io.File(outputPath).appendAll(result)
          }
        }
      }
    }



  }


}
