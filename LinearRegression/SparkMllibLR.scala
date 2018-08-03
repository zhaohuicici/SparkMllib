package SparkMllibOtheralgorithm


/**
  *　线性回归, 建立商品价格与消费者输入之间的关系,
  * 预测价格
  * https://my.oschina.net/sunmin/blog/719693
  */

import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.{SparkConf, SparkContext}

object SparkMllibLR  {
  val conf = new SparkConf()     //创建环境变量
    .setMaster("local")        //设置本地化处理
    .setAppName("LinearRegression")//设定名称
  val sc = new SparkContext(conf)  //创建环境变量实例

  def main(args: Array[String]) {
    val data = sc.textFile("./src/SparkMllibOtheralgorithm/lr.txt")//获取数据集路径
    val parsedData = data.map { line =>	 //开始对数据集处理
      val parts = line.split('|') //根据逗号进行分区
      LabeledPoint(parts(0).toDouble, Vectors.dense(parts(1).split(',').map(_.toDouble)))
    }.cache() //转化数据格式

    //LabeledPoint,　numIterations, stepSize
    val model = LinearRegressionWithSGD.train(parsedData, 2, 0.1) //建立模型

    val result = model.predict(Vectors.dense(1, 3))//通过模型预测模型
    println(model.weights)
    println(model.weights.size)
    println(result)	//打印预测结果
  }
}
